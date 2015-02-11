#include "./include/dataset.h"

#define eta 0.1

typedef struct rbm {
    int nvisible, nhidden;
    double **W;
    double *hbias, *vbias;
} rbm;


void init_model(rbm *m, int nvisible, int nhidden){
    double low, high; 
    int i, j;

    low = -4 * sqrt((double)6 / (nvisible + nhidden));
    high = 4 * sqrt((double)6 / (nvisible + nhidden));

    m->nhidden = nhidden;
    m->nvisible = nvisible;
    m->W = (double**)malloc(nhidden * sizeof(double*));
    for(i = 0; i < nhidden; i++){
        m->W[i] = (double*)malloc(nvisible * sizeof(double));
    }
    for(i = 0; i < nhidden; i++){
        for(j = 0; j < nvisible; j++){
            m->W[i][j] = random_double(low, high);
        }
    }
    m->vbias = (double*)calloc(nvisible, sizeof(double));
    m->hbias = (double*)calloc(nhidden, sizeof(double));
}

void free_model(rbm *m){
    int i;

    for(i = 0; i < m->nhidden; i++){
        free(m->W[i]);
    }
    free(m->W);
    free(m->vbias);
    free(m->hbias);

}

void get_hprob_given_vsample(const rbm *m, const double *vsample, double *hprob){
    int i, j;
    double s = 0;
    for(i = 0; i < m->nhidden; i++){
        s = 0;
        for(j = 0; j < m->nvisible; j++){
            s += m->W[i][j] * vsample[j]; 
        }
        hprob[i] = sigmoid(m->hbias[i] + s);
    }
}

void sample_h_from_hprob(const rbm *m, const double *hprob, double *hsample){
    int i;
    double u;
    for(i = 0; i < m->nhidden; i++){
        u = random_double(0.0, 1.0); 
        if(u < hprob[i])
            hsample[i] = 1.0;
        else
            hsample[i] = 0.0;
    }
}

void get_vprob_given_hsample(const rbm *m, const double *hsample, double *vprob){
    int i, j;
    double s = 0;
    for(i = 0; i < m->nvisible; i++){
        s = 0;
        for(j = 0; j < m->nhidden; j++){
            s += m->W[j][i] * hsample[j]; 
        }
        vprob[i] = sigmoid(m->vbias[i] + s);
    }
}

void sample_v_from_vprob(const rbm *m, const double *vprob, double *vsample){
    int i;
    double u;
    for(i = 0; i < m->nvisible; i++){
        u = random_double(0.0, 1.0);
        if(u < vprob[i])
            vsample[i] = 1.0;
        else
            vsample[i] = 0.0;
    }
}

void gibbs_sampling_hvh(const rbm *m, const double *hsample, int step, double *end_hprob, double *end_hsample, double *end_vprob, double *end_vsample){
    int k;

    memcpy(end_hsample, hsample, m->nhidden * sizeof(double));

    //printf("origin:\n", k);
    //print_hsample(m, end_hsample);

    for(k = 0; k < step; k++){
        get_vprob_given_hsample(m, end_hsample, end_vprob);
        sample_v_from_vprob(m, end_vprob, end_vsample);
        get_hprob_given_vsample(m, end_vsample, end_hprob);
        sample_h_from_hprob(m, end_hprob, end_hsample);

        //printf("epcho %d:\n", k);
        //print_hsample(m, end_hsample);
    }
}

void gibbs_sampling_vhv(const rbm *m, const double *vsample, int step, double *end_hprob, double *end_hsample, double *end_vprob, double *end_vsample){
    int k;

    memcpy(end_vsample, vsample, m->nvisible * sizeof(double));

    for(k = 0; k < step; k++){
        get_hprob_given_vsample(m, end_vsample, end_hprob);
        sample_h_from_hprob(m, end_hprob, end_hsample);
        get_vprob_given_hsample(m, end_hsample, end_vprob);
        sample_v_from_vprob(m, end_vprob, end_vsample);
#ifdef DEBUG
        if((k+1) % 100 == 0){
            printf("step :%d\n", k+1);
        }
#endif
    }
}

double get_free_energy(const rbm *m, const double *v){
    double vterm, hterm, s;
    int i, j; 
    for(i = 0, vterm = 0; i < m->nvisible; i++)
        vterm += v[i] * m->vbias[i];
    for(i = 0, hterm = 0; i < m->nhidden; i++){
        for(j = 0, s = 0.0; j < m->nvisible; j++){
            s += v[j] * m->W[i][j];
        }
        hterm += log(exp(m->hbias[i] + s) + 1.0);
    }
    return -vterm - hterm;
}

double get_pseudo_likelihood_cost(const rbm *m, dataset *d){
    int i, j;
    double old_val;
    double fe, fe_flip;
    double s;
    static int flip_idx = 0;

    for(i = 0, s = 0.0; i < d->N; i++){
        fe = get_free_energy(m, d->input[i]);

        old_val = d->input[i][flip_idx];
        d->input[i][flip_idx] = 1.0 - old_val;
        fe_flip = get_free_energy(m, d->input[i]);
        d->input[i][flip_idx] = old_val;

        s += m->nvisible * log(sigmoid(fe_flip - fe));
#ifdef DEBUG
        if((i+1) % 10000 == 0){
            printf("calculating pseudo likelihood cost! loop :%d\n", i+1);
        }
        fflush(stdout);
#endif
    }

    return s / d->N;
}

void print_model(const rbm *m, const dataset *d){
    printf("Weight:\n"); 
    int i, j;
    for(i = 0; i < m->nhidden; i++){
        printf("Hidden Node %d :\n", i);
        for(j = 0; j < m->nvisible; j++){
            printf("%lf%s", m->W[i][j], ((j+1) % d->ncol == 0 ? "\n" : "\t"));
        }
    }
    printf("VBias:\n");
    for(i = 0; i < m->nvisible; i++){
        printf("%lf%s", m->vbias[i], (i+1) % d->ncol == 0 ? "\n" : "\t");
    }
    printf("HBias:\n");
    for(i = 0; i < m->nhidden; i++){
        printf("%lf%s", m->hbias[i], (i+1) % 25 == 0 ? "\n" : "\t");
    }
}

void print_vsample(FILE *f, const rbm *m, const dataset *d, const double *vsample){
    int i;
    for(i = 0; i < m->nvisible; i++){
        fprintf(f, "%.5lf%s", vsample[i], (i+1) % d->ncol == 0 ? "\n" : "\t");
    }
}

void print_hsample(FILE *f, const rbm *m, const double *hsample){
    int i;
    for(i = 0; i < m->nhidden; i++){
        fprintf(f, "%.5lf%s", hsample[i], (i+1) % 25 == 0 ? "\n" : "\t");
    }
}

void dump_weight(FILE *f, const rbm *m, const dataset *d){
    int i, j, k;
    for(i = 0; i < 100; i++){
        fprintf(f, "Hidden Node %d\n", i);
        for(j = 0; j < m->nvisible; j++){
            fprintf(f, "%.5lf%s", m->W[i][j], (j+1) % d->ncol == 0 ? "\n" : "\t");
        }
        fflush(f);
    }
}

void print_sample(FILE *f, const rbm *m, const dataset *d, double **vsample, int sample_size){
    int i, j, k;
    double v2_sample[DEFAULT_MAXSIZE], v2_prob[DEFAULT_MAXSIZE], h2_sample[DEFAULT_MAXSIZE], h2_prob[DEFAULT_MAXSIZE];
    double **start_sample;
    int n_sample = 10;
    int step = 1000;

    start_sample = (double**)calloc(sample_size ,sizeof(double*));
    
    for(i = 0; i < n_sample; i++){
        fprintf(f, "sample %d\n", i);
        for(j = 0; j < sample_size; j++){
            if(start_sample[j] == NULL)
                start_sample[j] = vsample[j];
            gibbs_sampling_vhv(m, start_sample[j], step, h2_prob, h2_sample, v2_prob, v2_sample);
            print_vsample(f, m, d, v2_prob);
            start_sample[j] = v2_sample;
        }
    }

    free(start_sample);
}

int main(){
    int image_fd, label_fd;
    FILE *W_file, *V_sample_file, *log_file;
    uint32_t magic_n, N, nrow, ncol, data;
    int nvisible, nhidden;
    int mini_batch, training_epcho;
    time_t start_time, end_time, total_time = 0;
    int i, j, k, p, q;
    int epcho;
    rbm m;
    dataset d;
    uint8_t x;
    double v1_sample[DEFAULT_MAXSIZE], v1_prob[DEFAULT_MAXSIZE], h1_sample[DEFAULT_MAXSIZE], h1_prob[DEFAULT_MAXSIZE];
    double v2_sample[DEFAULT_MAXSIZE], v2_prob[DEFAULT_MAXSIZE], h2_sample[DEFAULT_MAXSIZE], h2_prob[DEFAULT_MAXSIZE];
    double delta_W[DEFAULT_MAXSIZE][DEFAULT_MAXSIZE], delta_vbias[DEFAULT_MAXSIZE], delta_hbias[DEFAULT_MAXSIZE];
    double *chain_start = NULL;
    rio_t rio_training_set_x;

    image_fd = open("../data/train-images-idx3-ubyte", O_RDONLY);
    W_file = fopen("weight.txt", "w");
    V_sample_file = fopen("sample.txt", "w");
    freopen("rbm.log", "w", stdout);

    if(image_fd == -1){
        fprintf(stderr, "cannot open file");
        exit(1);
    }

    rio_readinitb(&rio_training_set_x, image_fd, 0);

    read_uint32(&rio_training_set_x, &magic_n);
    read_uint32(&rio_training_set_x, &N);
    read_uint32(&rio_training_set_x, &nrow);
    read_uint32(&rio_training_set_x, &ncol);

#ifdef DEBUG
    printf("magic number: %u\nN: %u\nnrow: %u\nncol: %u\n", magic_n, N, nrow, ncol);
    fflush(stdout);
#endif

    init_dataset(&d, N, nrow, ncol);
    load_dataset_input(&rio_training_set_x, &d);
    close(image_fd);

    srand(1234);
    nvisible = d.nrow * d.ncol;
    nhidden = 500;
    mini_batch = 1;
    training_epcho = 15;
    init_model(&m, nvisible, nhidden);

    fprintf(W_file, "epcho 0\n");
    dump_weight(W_file, &m, &d);
    
    for(epcho = 0; epcho < training_epcho; epcho++){

        start_time = time(NULL);
        for(k = 0; k < d.N / mini_batch; k++){
            for(j = 0; j < m.nhidden; j++){
                delta_hbias[j] = 0;
                for(p = 0; p < m.nvisible; p++){
                    delta_W[j][p] = 0; 
                }
            }
            for(p = 0; p < m.nvisible; p++){
                delta_vbias[p] = 0; 
            }
#ifdef DEBUG
            if((k+1) % 300 == 0){
                printf("epcho:%d\tstep:%d\n", epcho + 1, k + 1);
            }
            fflush(stdout);
#endif

            for(i = 0; i < mini_batch; i++){

                memcpy(v1_sample, d.input[k*mini_batch+i], m.nvisible * sizeof(double));

                //printf("vsample:\n");
                //print_vsample(out_file, &m, v1_sample);

                get_hprob_given_vsample(&m, v1_sample, h1_prob);
                sample_h_from_hprob(&m, h1_prob, h1_sample);

                if(chain_start == NULL){
                    chain_start = h1_sample;
                }

                gibbs_sampling_hvh(&m, chain_start, 5, h2_prob, h2_sample, v2_prob, v2_sample);

                /**
                 * mini-batch调整delta
                 */
                for(j = 0; j < m.nhidden; j++){
                    delta_hbias[j] += h1_prob[j] - h2_prob[j];
                }
                for(j = 0; j < m.nvisible; j++){
                    delta_vbias[j] += v1_sample[j] - v2_sample[j];
                }
                for(j = 0; j < m.nhidden; j++){
                    for(p = 0; p < m.nvisible; p++){
                        delta_W[j][p] += h1_prob[j] * v1_sample[p] - h2_prob[j] * v2_sample[p];
                    }
                }

                chain_start = h2_sample;

                //printf("hprob:\n");
                //print_hsample(&m, h1_prob);
                //printf("hsample:\n");
                //print_hsample(&m, h1_sample);
            }

            /*
             * 根据delta调整参数
             */
            for(j = 0; j < m.nhidden; j++){
                for(p = 0; p < m.nvisible; p++){
                    m.W[j][p] += eta * delta_W[j][p] / mini_batch;
                }
            }
            for(j = 0; j < m.nhidden; j++){
                m.hbias[j] += eta * delta_hbias[j] / mini_batch;
            }
            for(j = 0; j < m.nvisible; j++){
                m.vbias[j] += eta * delta_vbias[j] / mini_batch;
            }

        }
#ifdef DEBUG
        printf("epcho %d cost: %.5lf\n", epcho + 1, get_pseudo_likelihood_cost(&m, &d));
#endif
        fprintf(W_file, "epcho %d\n", epcho + 1);
        dump_weight(W_file, &m, &d);
        end_time = time(NULL);
        printf("epcho %d time : %.2f min\n", epcho + 1, (float)(end_time - start_time) / 60);
        fflush(stdout);
        total_time += end_time - start_time;
    }

    print_sample(V_sample_file, &m, &d, d.input + 100, 20);
    printf("total time : %.2f min\n", (float)(total_time) / 60);

    //print_dataset(&d);

    //print_model(&m, &d);

    fclose(log_file);
    fclose(W_file);
    fclose(V_sample_file);
    free_dataset(&d);
    free_model(&m);
    return 0;
}
