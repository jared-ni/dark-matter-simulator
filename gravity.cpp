#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <chrono>
#include <x86intrin.h>
#include "CImg/CImg.h"
#include <omp.h>
#include <random>
#include <papi.h>
#include <cassert>


// Set to `true` to output video frames (which can be passed to ffmpeg to produce video)
#define OUTPUT_VIDEO_FRAMES true
// Set to `true` to output total momentum at regular intervals to check correctness
#define CHECK_MOMENTUM false
// Set to `true` to run PAPI benchmarks
#define BENCHMARK_PAPI false
// Choose what to benchmark with PAPI
const std::string PAPI_TYPE_DESCRIPTION = "PAPI_VEC_DP";
const int PAPI_TYPE =
    (PAPI_TYPE_DESCRIPTION == "PAPI_DP_OPS") ? PAPI_DP_OPS :
    (PAPI_TYPE_DESCRIPTION == "PAPI_LST_INS") ? PAPI_LST_INS :
    (PAPI_TYPE_DESCRIPTION == "PAPI_VEC_DP") ? PAPI_VEC_DP :
    -1;


// One time step
const double DELTA_T = 0.001;
// Number of time steps that elapse between each frame of video
const int STEPS_PER_WRITE = 100;
// Value of the gravitational constant
const int GRAVITATIONAL_CONST = 1;

// Width and height of image frames that make up video
// (If you change this, you will also need to modify the `ffmpeg` command when
//  generating the video)
const int WIDTH = 1024;
const int HEIGHT = 768;

// A 2D projection of the 3D universe is made for viewing.
// A camera is placed at (`CAMERA_X`, `CAMERA_Y`, `CAMERA_Z`).
// A line is drawn from the camera to each point in the universe, and the
// point at which this line intersects the xy-plane is taken as the 2D projection.
const int CAMERA_X = WIDTH / 2;
const int CAMERA_Y = HEIGHT / 2;
const int CAMERA_Z = std::max(WIDTH, HEIGHT);
// `visible_to_camera` defines the direction in which the camera is pointing
// and thus in which objects in the universe are visible.
// The minus 1 prevents objects too close to the camera from causing errors.
inline bool visible_to_camera(double z) {return z < (double) CAMERA_Z-1;}

// Image on which the projection of the universe is drawn and displayed
cimg_library::CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3);
std::random_device rd;
std::mt19937 gen(rd());



// 3D vector, with x-, y-, and z-coordinates
struct vec {
    double x;
    double y;
    double z;
    vec();
    vec(double, double, double);
};

vec::vec() {}
vec::vec(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

// Vector addition using SIMD
vec operator+(vec u, vec v) {
    // Load u and v into SIMD registers
    __m256d simd_u = _mm256_set_pd(0.0, u.z, u.y, u.x);
    __m256d simd_v = _mm256_set_pd(0.0, v.z, v.y, v.x);
    // Perform addition using SIMD
    __m256d result = _mm256_add_pd(simd_u, simd_v);  
    // Return the result as a vec
    vec res;
    _mm256_storeu_pd(&res.x, result);
    return res;
}

// Vector subtraction using SIMD
vec operator-(vec u, vec v) {
    __m256d simd_u = _mm256_set_pd(0.0, u.z, u.y, u.x);
    __m256d simd_v = _mm256_set_pd(0.0, v.z, v.y, v.x);
    __m256d result = _mm256_sub_pd(simd_u, simd_v);
    vec res;
    _mm256_storeu_pd(&res.x, result);
    return res;
}

// Scalar multiplication using SIMD
vec operator*(double a, vec v) {
    __m256d simd_v = _mm256_set_pd(0.0, v.z, v.y, v.x);
    // Broadcast scalar 'a' into all elements of a SIMD register
    __m256d simd_a = _mm256_set1_pd(a);
    __m256d result = _mm256_mul_pd(simd_a, simd_v);
    vec res;
    _mm256_storeu_pd(&res.x, result);
    return res;
}

// Scalar multiplication from the right (same as from left)
vec operator*(vec v, double a) {
    return a * v;
}

// Scalar multiplication by reciprocal from the right
vec operator/(vec v, double a) {
    return 1/a * v;
}

// Dot product using SIMD for parallel multiplications
double operator*(vec u, vec v) {
    __m256d simd_u = _mm256_set_pd(0.0, u.z, u.y, u.x);
    __m256d simd_v = _mm256_set_pd(0.0, v.z, v.y, v.x);
    __m256d mul_result = _mm256_mul_pd(simd_u, simd_v);
    
    // Horizontal addition of the elements
    __m256d sum1 = _mm256_hadd_pd(mul_result, mul_result);
    __m128d sum2 = _mm256_extractf128_pd(sum1, 1);
    __m128d sum = _mm_add_pd(_mm256_castpd256_pd128(sum1), sum2);
    
    // Extracting the result from the sum
    double result;
    _mm_store_sd(&result, sum);
    
    return result;
}

// Update with vector addition
void operator+=(vec& u, vec v) {
    u = u + v;
}

// Represents an object in the universe
struct object {
    double mass;  // Mass
    double radius;  // Radius (object is a sphere) (used only for display)
    vec pos;  // Position
    vec vel;  // Velocity
    vec acc;  // Acceleration

    int CUR_COLOR[3];  // Color of object (used only for display)

    void calculate_acc(object[], int);
    void update_pos_and_vel();
    void draw();
    object();
};

object::object() {
    std::uniform_int_distribution<> dis(1, 100);
    int randomNumber = dis(gen);

    if (randomNumber < 20) {
        CUR_COLOR[0] = 205;
        CUR_COLOR[1] = 0;
        CUR_COLOR[2] = 0;
    } else if (randomNumber < 40) {
        CUR_COLOR[0] = 255;
        CUR_COLOR[1] = 128;
        CUR_COLOR[2] = 0;
    } else if (randomNumber < 60) {
        CUR_COLOR[0] = 255;
        CUR_COLOR[1] = 255;
        CUR_COLOR[2] = 204;
    } else if (randomNumber < 70) {
        CUR_COLOR[0] = 204;
        CUR_COLOR[1] = 255;
        CUR_COLOR[2] = 255;
    } else if (randomNumber < 80) {
        CUR_COLOR[0] = 153;
        CUR_COLOR[1] = 255;
        CUR_COLOR[2] = 255;
    } else if (randomNumber < 90) {
        CUR_COLOR[0] = 0;
        CUR_COLOR[1] = 255;
        CUR_COLOR[2] = 255;
    } else {
        CUR_COLOR[0] = 255;
        CUR_COLOR[1] = 255;
        CUR_COLOR[2] = 255;
    }
}


// Calculate acceleration of this object based on the gravitational effects
// of all `num_objects` of the objects in `objects`
void object::calculate_acc(object objects[], int num_objects) {
    // Net acceleration on this object due to gravitational effects of all other objects
    this->acc = vec(0, 0, 0);

    for (int i = 0; i < num_objects; ++i) {
        if (this == &objects[i]) {
            // Object should ignore itself
            continue;
        }
        // Displacement and distance from this object to object `i`
        vec displacement = objects[i].pos - this->pos;
        double squared_distance = displacement * displacement;
        double distance = sqrt(squared_distance);
        // Magnitude of acceleration caused by object `i` by Newton's law of gravitation
        double acc_magnitude = GRAVITATIONAL_CONST * objects[i].mass / squared_distance;
        // Unit vector in the direction of object `i`
        vec acc_direction = displacement / distance;
        // Calculate acceleration vector and update net acceleration
        this->acc += acc_magnitude * acc_direction;
    }
}

// Update velocity and position of this object
void object::update_pos_and_vel() {
    this->vel += this->acc * DELTA_T;
    this->pos += this->vel * DELTA_T;
}

// Draw this object on `image` based on 2D projection:
// a line is drawn from the camera to each point, and the intersection of that
// line with the xy-plane is used as the 2D projection
void object::draw() {
    if (!visible_to_camera(this->pos.z)) return;
    double x_proj =
        CAMERA_X + (this->pos.x - CAMERA_X) * CAMERA_Z / (CAMERA_Z - this->pos.z);
    double y_proj =
        CAMERA_Y + (this->pos.y - CAMERA_Y) * CAMERA_Z / (CAMERA_Z - this->pos.z);
    double radius_proj = this->radius * CAMERA_Z / (CAMERA_Z - this->pos.z);

    image.draw_circle(x_proj, y_proj, radius_proj, this->CUR_COLOR);
}


// Represents the universe
struct universe {
    int num_objects;  // Number of objects in universe
    object *objects;  // Array of all objects in universe
    int start_idx, end_idx;  // Indicates subarray of objects assigned to this rank
    universe(int, int, int);
    universe();
    ~universe();
    void update_state();
    void draw();
    vec calculate_momentum();
};

// Constructor: initialize universe with enough dynamically-allocated
// memory for `num_objects` objects, and determine subarray that this rank
// is responsible for
universe::universe(int num_objects, int start_idx, int end_idx) {
    this->num_objects = num_objects;
    this->objects = new object[num_objects];
    this->start_idx = start_idx;
    this->end_idx = end_idx;
}
universe::universe() {
    this->objects = nullptr;
}

// Destructor: Free dynamically-allocated memory
universe::~universe() {
    delete[] objects;
}

// Update state of assigned objects
void universe::update_state() {
    #pragma omp parallel for
    for (int i = this->start_idx; i < this->end_idx; ++i) {
        // std::cerr << "Number of threads: " << omp_get_num_threads( ) << std::endl;
        this->objects[i].calculate_acc(this->objects, this->num_objects);
    }
    
    #pragma omp parallel for
    for (int i = this->start_idx; i < this->end_idx; ++i) {
        this->objects[i].update_pos_and_vel();
    }
}

// Draw all objects on `image` and output the image to stdout
void universe::draw() {
    // Reset image with all black pixels
    image.fill(0);
    // Draw each object on image
    for (int i = 0; i < this->num_objects; ++i) {
        objects[i].draw();
    }
    // Output pixels of image
    char *s;
    s = reinterpret_cast<char*>(image.data() + WIDTH*HEIGHT);  // Green
    std::cout.write(s, WIDTH*HEIGHT);
    s = reinterpret_cast<char*>(image.data() + 2*WIDTH*HEIGHT);  // Blue
    std::cout.write(s, WIDTH*HEIGHT);
    s = reinterpret_cast<char*>(image.data());  // Red
    std::cout.write(s, WIDTH*HEIGHT);
}

// Calculate total momentum of objects in universe
vec universe::calculate_momentum() {
    vec momentum(0, 0, 0);
    for (int i = 0; i < this->num_objects; ++i) {
        momentum += this->objects[i].mass * this->objects[i].vel;
    }
    return momentum;
}


int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided != MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Warning MPI did not provide MPI_THREAD_FUNNELED\n");
    }
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::chrono::time_point<std::chrono::steady_clock> start_time;
    if (rank == 0) {
        start_time = std::chrono::steady_clock::now();
        std::cerr << "Number of ranks: " << num_ranks << std::endl;
    }

    #if BENCHMARK_PAPI
        int event_set = PAPI_NULL;
        int r0 = PAPI_library_init(PAPI_VER_CURRENT);
        int r1 = PAPI_create_eventset(&event_set);
        int r2 = PAPI_add_event(event_set, PAPI_TYPE);
        int r3 = PAPI_start(event_set);
        assert(r0 == PAPI_VER_CURRENT);
        assert(r1 == 0 && r2 == 0 && r3 == 0);
    #endif

    // Get initial state of universe
    universe global_univ;
    int num_time_steps, num_objects;
    std::ifstream fin("initial_state.txt");
    fin >> num_time_steps >> num_objects;
    if (rank == 0) {
        global_univ.num_objects = num_objects;
        global_univ.objects = new object[num_objects];
        for (int i = 0; i < num_objects; ++i) {
            fin >> global_univ.objects[i].mass >> global_univ.objects[i].radius
                >> global_univ.objects[i].pos.x >> global_univ.objects[i].pos.y >> global_univ.objects[i].pos.z
                >> global_univ.objects[i].vel.x >> global_univ.objects[i].vel.y >> global_univ.objects[i].vel.z;
        }
    }
    fin.close();

    // Determine assignment for each rank
    int assigned_idxs[num_ranks+1];
    assigned_idxs[0] = 0;
    for (int r = 0; r < num_ranks; ++r) {
        int assigned_len = num_objects / num_ranks;
        if (r < num_objects % num_ranks) {
            ++assigned_len;
        }
        assigned_idxs[r+1] = assigned_idxs[r] + assigned_len;
    }

    // Each rank has its own `univ` struct, with an assigned piece of work
    universe univ(num_objects, assigned_idxs[rank], assigned_idxs[rank+1]);

    // Run simulation -- repeat for each time step
    for (int t = 0; t < num_time_steps; ++t) {
        // Root distributes work and draws image
        MPI_Request send_reqs[num_ranks];
        if (rank == 0) {
            for (int r = 0; r < num_ranks; ++r) {
                // Send `global_univ.objects` to each rank `r`
                int bytes = num_objects * sizeof(object);
                MPI_Isend(global_univ.objects, bytes, MPI_BYTE, r,
                          0, MPI_COMM_WORLD, &send_reqs[r]);
            }
            if (t % STEPS_PER_WRITE == 0) {
                #if OUTPUT_VIDEO_FRAMES
                    global_univ.draw();
                #endif
                #if CHECK_MOMENTUM
                    vec momentum = global_univ.calculate_momentum();
                    std::cerr << "Momentum: ("
                            << momentum.x << ", " << momentum.y << ", " << momentum.z
                            << ")" << std::endl;
                #endif
            }
        }

        // Receive state of universe from root
        int bytes = num_objects * sizeof(object);
        MPI_Recv(univ.objects, bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Update state of the assigned objects
        univ.update_state();

        // Send assigned updates to root
        MPI_Request local_send_req;
        bytes = (univ.end_idx - univ.start_idx) * sizeof(object);
        MPI_Isend(&univ.objects[univ.start_idx], bytes, MPI_BYTE, 0, 0,
                  MPI_COMM_WORLD, &local_send_req);

        if (rank == 0) {
            MPI_Waitall(num_ranks, send_reqs, MPI_STATUSES_IGNORE);

            // Root receives updates from each rank
            MPI_Request recv_reqs[num_ranks];
            for (int r = 0; r < num_ranks; ++r) {
                int bytes = (assigned_idxs[r+1] - assigned_idxs[r]) * sizeof(object);
                MPI_Irecv(&global_univ.objects[assigned_idxs[r]], bytes, MPI_BYTE,
                          r, 0, MPI_COMM_WORLD, &recv_reqs[r]);
            }

            MPI_Waitall(num_ranks, recv_reqs, MPI_STATUSES_IGNORE);
        }

        MPI_Wait(&local_send_req, MPI_STATUS_IGNORE);
    }

    #if BENCHMARK_PAPI
        long long int counter;
        int r4 = PAPI_stop(event_set, &counter);
        assert(r4 == 0);
        std::cerr << "(rank " << rank << ") "
                  << PAPI_TYPE_DESCRIPTION << ": " << counter << std::endl;
    #endif

    if (rank == 0) {
        auto end_time = std::chrono::steady_clock::now();
        double runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;
        std::cerr << "Runtime (seconds): " << runtime << std::endl;
    }

    MPI_Finalize();
}
