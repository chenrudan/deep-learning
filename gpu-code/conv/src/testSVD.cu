/// filename: testSVD.cu
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cstdio>
#include <string>

using namespace std;

double* mallocOnGpu(const int len){
	double* data;
	try{
		cudaError_t status;
		status = cudaMalloc((void**) &data, \
			len * sizeof(double));
		throw status;
	}
	catch(cudaError_t status){
		if (status != cudaSuccess) {
			fprintf(stderr, "!!!! device memory allocation error\n");
			exit(EXIT_FAILURE);
		}
	}
	return data;
}

__global__ void kCropImg(const double* ori_img, double* dst_img, \
		const int row_start, const int cropped_height, \
		const int col_start, const int cropped_width, \
		const int ori_width){

	int idx = threadIdx.x;

	while (idx < cropped_height*cropped_width) {
		int ori_row_idx = idx / cropped_width + row_start;
		int ori_col_idx = idx % cropped_width + col_start;
		dst_img[idx] = ori_img[ori_row_idx*ori_width + ori_col_idx];
		idx += blockDim.x;
	}
	__syncthreads();
}

__global__ void kComputeNorm(const double* vec, double* norm, const int len){
	//每一个block计算一个模
	extern __shared__ double sh_norm[];

	int pow2_len = len;
	if (pow2_len & (pow2_len - 1)) {
		while (pow2_len & (pow2_len - 1)){
			pow2_len &= pow2_len - 1;
		}
	}

	int i = threadIdx.x;
	while (i < len) {
		sh_norm[i] = vec[i]*vec[i];
		i += blockDim.x;
	}

	int reduce_len = pow2_len > blockDim.x ? blockDim.x : pow2_len;
	int times = len / reduce_len;

	int vec_pos = threadIdx.x + reduce_len * times;
	if (vec_pos > (reduce_len * times) && vec_pos < len) {
		sh_norm[vec_pos - reduce_len] += sh_norm[vec_pos];
	}
	__syncthreads();

	for (int j = times-1; j >= 0; j--) {
		vec_pos = threadIdx.x + j*reduce_len;
		if (threadIdx.x == 0 && (j + 1) * reduce_len < len) {
			sh_norm[0] += sh_norm[(j + 1) * reduce_len];
		}
		__syncthreads();
		for (int active_thread = (reduce_len >> 1); active_thread; active_thread >>= 1) {
			if (threadIdx.x < active_thread) {
				sh_norm[vec_pos] += sh_norm[vec_pos + active_thread];
			}
			__syncthreads();
		}
	}

	if (threadIdx.x == 0) {
		norm[0] = sqrt(sh_norm[0]);
	}

	__syncthreads();
}

//dst = (src + [added_value, 0, ..., 0]) * scale
__global__ void kComputeHouseholderVec(const double* src, double* dst, \
		double added_value, double scale, const int len) {
	int idx = threadIdx.x;
	while (idx < len) {
		if (idx == 0) {
			dst[idx] = scale * (src[idx] + added_value);
		} else
			dst[idx] = scale * src[idx];
		idx += blockDim.x;
	}
}


void showValue(double* vec, const int len) {
	double* vec_cpu = new double[len];
	cudaMemcpy(vec_cpu, vec, sizeof(double)*len, cudaMemcpyDeviceToHost);
	for (int k = 0; k < len; ++k) {
		cout << vec_cpu[k] << " ";
	}
	cout << endl;
	delete[] vec_cpu;
}

void showValue(double* mat, const int height, const int width, string name = "") {
	cout << "========="<< name << "=========" << endl;
	cout << height << ":" << width << endl;
	double* mat_cpu = new double[width*height];
	cudaMemcpy(mat_cpu, mat, sizeof(double)*width*height, cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			cout << mat_cpu[i*width + j] << " ";
		}
		cout << endl;
	}
	delete[] mat_cpu;
}

double getGpuValue(const double* gpu_value, int pos){
	double tmp;
	cudaMemcpy(&tmp, gpu_value + pos, sizeof(double), cudaMemcpyDeviceToHost);
	return tmp;
}

double getFirstGpuValue(const double* gpu_value){
	return getGpuValue(gpu_value, 0);
}

void computeHouseHolderVecAndAlpha(const double* cropped_A, const int vec_len, \
		double *householder_vector_gpu, double& alpha_cpu, double& sigma_gpu){
	double y1_u = 0;
	double* u_norm = mallocOnGpu(1);
	kComputeNorm<<<1, 512, sizeof(double)*vec_len>>>(cropped_A, \
		u_norm, vec_len);

	y1_u = getFirstGpuValue(cropped_A);
	alpha_cpu = y1_u > 0 ? -getFirstGpuValue(u_norm) : getFirstGpuValue(u_norm);
	sigma_gpu = (y1_u - alpha_cpu) / (-alpha_cpu);

	kComputeHouseholderVec<<<1, 1024>>>(cropped_A, householder_vector_gpu, \
			-alpha_cpu, 1/(y1_u - alpha_cpu), vec_len);

	cudaFree(u_norm);

}

int main(){

	//只针对一张图处理
	int img_size = 100;
	double* ori_img = mallocOnGpu(img_size*img_size);

	double* x_cpu = new double[img_size*img_size];
	for (int i = 0; i < img_size; ++i) {
		for (int j = 0; j < img_size; ++j) {
			x_cpu[i*img_size + j] = 1;
		}
	}
	cudaMemcpy(ori_img, x_cpu, sizeof(double)*img_size*img_size, cudaMemcpyHostToDevice);

	//alpha是对角化后左上角的元素值，anti_alpha是用来计算矢量u
	double alpha = 0;
	double sigma_u = 0;
	//beta对角化后第一行第二个值
	double beta = 0;
	double sigma_v = 0;

	//假设从第一行开始，下标为0，截取出来的矩阵是A(0:m-1,0)，一个列向量
	int crop_idx = 9;
	int crop_col_u_start = crop_idx;
	int crop_u_height = img_size - crop_idx;
	double* cropped_A_for_u = mallocOnGpu(crop_u_height );
	double* householder_vector_u = mallocOnGpu(crop_u_height );

	double scale_one = 1;
	double scale_minus_one = -1;
	double scale_tar = 0;
	kCropImg<<<1, 1024>>>(ori_img, cropped_A_for_u, crop_idx, \
			crop_u_height, crop_col_u_start, 1, img_size);

	computeHouseHolderVecAndAlpha(cropped_A_for_u, crop_u_height , \
	          householder_vector_u, alpha, sigma_u);

	//求v的是个行矢量，使用的时候当成列向量
	int crop_col_v_start = crop_idx + 1;
	int crop_v_width = img_size - crop_col_v_start;
	double* cropped_A_for_v = mallocOnGpu(crop_v_width);
	double* householder_vector_v = mallocOnGpu(crop_v_width);
	kCropImg<<<1, 1024>>>(ori_img, cropped_A_for_v, crop_idx, \
			1, crop_col_v_start, crop_v_width, img_size);
	computeHouseHolderVecAndAlpha(cropped_A_for_v, crop_v_width, \
	          householder_vector_v, beta, sigma_v);

	double *cropped_A_for_z = mallocOnGpu(crop_u_height*crop_v_width);
	kCropImg<<<1, 1024>>>(ori_img, cropped_A_for_z, crop_idx, \
			crop_u_height, crop_col_v_start, crop_v_width, img_size);

	double *x = mallocOnGpu(crop_v_width);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, crop_v_width, \
	      crop_u_height, &sigma_u, householder_vector_u, 1, \
		  cropped_A_for_z, crop_v_width, &scale_tar, x, 1);

	double* z = mallocOnGpu(crop_v_width);
	double x_mutliply_v = 0;
	cublasDdot(handle, crop_v_width, x, 1, householder_vector_v, 1, &x_mutliply_v);
	double scale_xv = -x_mutliply_v*sigma_v;
	cublasDcopy(handle, crop_v_width, x, 1, z, 1);
	cublasDaxpy(handle, crop_v_width, &scale_xv, householder_vector_v, 1, z, 1);
/*
	cout << scale_xv << endl;
	showValue(cropped_A_for_z, crop_u_height, crop_v_width, "u_multiply_z");
	showValue(householder_vector_u, crop_u_height, 1, "householder_vector_v");
	showValue(x, crop_v_width, 1, "x");
	showValue(z, crop_v_width, 1, "z");
 */

	double *w = mallocOnGpu(crop_u_height);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, crop_u_height, \
	      crop_v_width, &sigma_v, householder_vector_v, 1, \
		  cropped_A_for_z, crop_v_width, &scale_tar, w, 1);

	//u*z
	double *delta_A = mallocOnGpu(crop_u_height*crop_v_width);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, crop_v_width, crop_u_height,
				1, &scale_one, z, crop_v_width, \
		  householder_vector_u, 1, &scale_tar, delta_A, crop_v_width);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, crop_v_width, crop_u_height,
				1, &scale_one, householder_vector_v, crop_v_width, \
		  w, 1, &scale_one, delta_A, crop_v_width);
	cublasDaxpy(handle, crop_v_width*crop_u_height, &scale_minus_one, \
	          delta_A, 1, cropped_A_for_z, 1);

	double *h = mallocOnGpu(crop_u_height*crop_u_height);
	double *g = mallocOnGpu(crop_v_width*crop_v_width);



	cout << sigma_v << endl;
	showValue(z, crop_v_width, 1, "z");
	showValue(householder_vector_u, crop_u_height, 1, "householder_vector_u");
	showValue(w, crop_v_width, 1, "w");
	showValue(householder_vector_v, crop_v_width, 1, "householder_vector_v");
	showValue(delta_A, crop_u_height, crop_v_width, "delta_A");
	showValue(cropped_A_for_z, crop_u_height, crop_v_width, "cropped_A_for_z");

	cudaFree(cropped_A_for_u);
	cudaFree(householder_vector_u);
	cudaFree(cropped_A_for_v);
	cudaFree(householder_vector_v);
	cudaFree(ori_img);
	cudaFree(cropped_A_for_z);
	cudaFree(x);

/*
  	//L最大是32
	int l_block_size = 100;

	int row_idx = 0;
	int col_idx = 0;
	for(int i = 1; i <= ceil(img_size / l_block_size); i++) {
		//下标从0开始
		row_idx = l_block_size * (i - 1);
		col_idx = row_idx;
		//A在此次中参与计算的列部分是A(row_idx:m, col_idx)
	}
*/


	return 0;
}


