
//#include "svd.hpp"
using namespace std;

template <typename Dtype>
SVD<Dtype>::SVD(Matrix<Dtype> *A, const int m, const int n) \
        : _height(m), _width(n), _alpha(0), \
        _sigma_u(0), _beta(0), _sigma_v(0), _scale_one(1), \
        _scale_minus_one(-1), _scale_zero(0){

    _A = new Matrix<Dtype>(m, n);
    _A->copyFromDevice(A);

    _householder_mat_p = new Matrix<Dtype>(n, n);
    _householder_mat_q = new Matrix<Dtype>(m, m);
    _householder_mat_p->reValue(1.0f);
    _householder_mat_q->reValue(1.0f);

    _householder_vec_u = new Matrix<Dtype>(m, 1);
    _householder_vec_v = new Matrix<Dtype>(n, 1);

    _w = new Matrix<Dtype>(m, 1);
    _z = new Matrix<Dtype>(n, 1);
    _x = new Matrix<Dtype>(n, 1);

    _h = new Matrix<Dtype>(m, m);
    _g = new Matrix<Dtype>(n, n);

    _cropped_A_for_u_v = new Matrix<Dtype>(m, 1);
    _cropped_A_for_z_w = new Matrix<Dtype>(m, n);

    _delta_A_for_A = new Matrix<Dtype>(m-1, n-1);


    cublasCreate(&handle);
}

template <typename Dtype>
SVD<Dtype>::~SVD(){
    delete _householder_mat_p;
    delete _householder_mat_q;

    delete _householder_vec_u;
    delete _householder_vec_v;

    delete _w;
    delete _z;
    delete _x;

    delete _h;
    delete _g;

    delete _cropped_A_for_u_v;
    delete _cropped_A_for_z_w;
//    cublasDestory(&handle);
}

template <typename Dtype>
void SVD<Dtype>::computeHouseHolderVecU(const int vec_start_idx){
   	_vec_start_idx = vec_start_idx;
	_vec_u_len = _height - vec_start_idx;
    _vec_v_len = _width - vec_start_idx - 1;

    _A->cropMatToNew(_cropped_A_for_u_v, _vec_start_idx, \
			_vec_u_len, _vec_start_idx, 1);

    _A->cropMatToNew(_cropped_A_for_z_w, _vec_start_idx, \
			_vec_u_len, _vec_start_idx+1, _vec_v_len);

    computeHouseHolderVecAndAlpha(_vec_u_len, \
         _householder_vec_u, _alpha, _sigma_u);

}

template <typename Dtype>
void SVD<Dtype>::computeHouseHolderVecV(){
    if (_vec_v_len <= 0) {
        return;
    }
    _A->cropMatToNew(_cropped_A_for_u_v, _vec_start_idx, 1, \
          _vec_start_idx+1, _vec_v_len);
    computeHouseHolderVecAndAlpha(_vec_v_len, \
         _householder_vec_v, _beta, _sigma_v);
}

template <typename Dtype>
void SVD<Dtype>::computeHouseHolderVecAndAlpha(const int vec_len, \
		Matrix<Dtype> *householder_vector_gpu, Dtype &alpha_cpu, \
        Dtype &sigma_gpu){
    Dtype u_norm = _cropped_A_for_u_v->computeNorm(vec_len);
    Dtype y1_u = _cropped_A_for_u_v->getFirstPosValue();

    alpha_cpu = y1_u > 0 ? -u_norm : u_norm;
    sigma_gpu = (y1_u - alpha_cpu) / (-alpha_cpu);
    kComputeHouseholderVec<<<1, 1024>>>(_cropped_A_for_u_v->getDevData(), \
            householder_vector_gpu->getDevData(), \
			-alpha_cpu, 1/(y1_u - alpha_cpu), vec_len);
}


//更新A(i:m, i)
template <typename Dtype>
void SVD<Dtype>::eliminateAForV() {
    _A->setValueAt(_vec_start_idx, _vec_start_idx, _alpha);
	for(int i = _vec_start_idx+1; i < _height; i++) {
		_A->setValueAt(i, _vec_start_idx, 0);
	}
}

//更新A(i, i+1:n)
template <typename Dtype>
void SVD<Dtype>::eliminateAForU() {
	_A->setValueAt(_vec_start_idx, _vec_start_idx+1, _beta);
	for(int i = _vec_start_idx+2; i < _width; i++) {
		_A->setValueAt(_vec_start_idx, i, 0);
	}
}

template <typename Dtype>
void SVD<Dtype>::computeHAndUpdateQ() {
//    _householder_vec_u->showValue("u_before");
    if (_vec_u_len < _height) {
        for (int i = _height-1; i >= 0; i--) {
            if (i < _vec_start_idx)
                _householder_vec_u->setValueAt(i, 0, 0);
            else
                _householder_vec_u->setValueAt(i, 0, \
                 _householder_vec_u->getPosValue(i - _vec_start_idx));
        }
    }
//       _householder_vec_u->showValue("u_after");
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _height, \
				_height, 1, &_sigma_u, \
				_householder_vec_u->getDevData(), _height, \
		        _householder_vec_u->getDevData(), 1, &_scale_zero, \
                _h->getDevData(), _height);
    // Hi = I - sigma_u * u * u'
    _h->subedByUnitMat();

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _height, \
				_height, _height, &_scale_one, \
				_householder_mat_q->getDevData(), _height, \
		        _h->getDevData(), _height, &_scale_zero, \
                _householder_mat_q->getDevData(), _height);

}

template <typename Dtype>
void SVD<Dtype>::computeGAndUpdateP() {
//    _householder_vec_v->showValue("v_before");
    for(int i = _width-1; i >= 0; i--) {
        if (i < _vec_start_idx+1)
            _householder_vec_v->setValueAt(i, 0, 0);
        else
            _householder_vec_v->setValueAt(i, 0, \
                 _householder_vec_v->getPosValue(i-_vec_start_idx-1));
    }
//    _householder_vec_v->showValue("v_after");
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _width, \
				_width, 1, &_sigma_v, \
				_householder_vec_v->getDevData(), _width, \
		        _householder_vec_v->getDevData(), 1, &_scale_zero, \
                _g->getDevData(), _width);
    // Hi = I - sigma_u * u * u'

    _g->subedByUnitMat();

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _width, \
				_width, _width, &_scale_one, \
				_g->getDevData(), _width, \
		        _householder_mat_p->getDevData(), _width, &_scale_zero, \
                _householder_mat_p->getDevData(), _width);
}

template <typename Dtype>
void SVD<Dtype>::updateA(){

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _vec_v_len, \
				_vec_u_len, 1, &_scale_minus_one, \
				_z->getDevData(), _vec_v_len, \
		        _householder_vec_u->getDevData(), 1, &_scale_zero, \
                _delta_A_for_A->getDevData(), _vec_v_len);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _vec_v_len, \
				_vec_u_len, 1, &_scale_minus_one, \
				_householder_vec_v->getDevData(), _vec_v_len, \
		        _w->getDevData(), 1, &_scale_one, \
                _delta_A_for_A->getDevData(), _vec_v_len);

//    _A->showValue("a_before");
    _A->subPortion(_delta_A_for_A, _vec_u_len-1, _vec_v_len);
 //   _A->showValue("a_after");
}

//w = sigma_v*A(i:m, i+1:n)*v
template <typename Dtype>
void SVD<Dtype>::computeW() {
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, \
				_vec_u_len, _vec_v_len, &_sigma_v, \
				_householder_vec_v->getDevData(), 1, \
		        _cropped_A_for_z_w->getDevData(), _vec_v_len, \
                &_scale_zero, _w->getDevData(), 1);
}

//z = x' - sigma_v*(x'*v)*v'
//x = sigma_u*A(i:m, i+1:n)'*u
template <typename Dtype>
void SVD<Dtype>::computeZ() {

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, \
				_vec_v_len, _vec_u_len, &_sigma_u, \
				_householder_vec_u->getDevData(), 1, \
		        _cropped_A_for_z_w->getDevData(), _vec_u_len, \
                &_scale_zero, _x->getDevData(), 1);

    Matrix<Dtype>* x_multi_v = new Matrix<Dtype>(1, 1);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, \
				1, _vec_v_len, &_scale_one, \
				_householder_vec_v->getDevData(), 1, \
		        _x->getDevData(), _vec_v_len, \
                &_scale_zero, x_multi_v->getDevData(), 1);
    Dtype x_multi_v_cpu = x_multi_v->getFirstPosValue();
    x_multi_v_cpu *= -_sigma_v;

    cublasDcopy(handle, _vec_v_len, _x->getDevData(), 1, \
              _z->getDevData(), 1);
    cublasDaxpy(handle, _vec_v_len, &x_multi_v_cpu, \
          _householder_vec_v->getDevData(), 1, _z->getDevData(), 1);
}

template <typename Dtype>
Matrix<Dtype>* SVD<Dtype>::getPAQ(Matrix<Dtype> *A) {
    Matrix<Dtype> *B = new Matrix<Dtype>(_height, _width);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _height, \
				_width, _height, &_scale_one, \
				A->getDevData(), _height, \
		        _householder_mat_q->getDevData(), _height, \
                &_scale_zero, B->getDevData(), _height);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _width, \
				_height, _width, &_scale_one, \
				_householder_mat_p->getDevData(), _width, \
		        B->getDevData(), _width, \
                &_scale_zero, B->getDevData(), _width);
    return B;
}

