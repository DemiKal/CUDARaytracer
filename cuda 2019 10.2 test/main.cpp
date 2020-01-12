#define TINYOBJLOADER_IMPLEMENTATION
#include  "pch.h"

__device__ bool hit_triangle(const ray& r, Triangle& tri) {
	//	Triangle tri({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 });

	const vec3 orig = r.GetOrigin();
	const vec3 dir = r.GetDirection();
	const vec3 move(0, 0, 0);
	vec3 v0 = move + tri.GetA();
	vec3 v1 = move + tri.GetB();
	vec3 v2 = move + tri.GetC();
	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;
	vec3 pvec = cross(dir, v0v2);
	float det = dot(v0v1, pvec);

	float invDet = 1 / det;

	vec3 tvec = orig - v0;

	float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	vec3 qvec = cross(tvec, v0v1);
	float v = dot(dir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	float t = dot(v0v2, qvec) * invDet;
	return t > 0;

}

__device__ vec3 color(const ray& r, Triangle* triangles, int triSize) {

	if (hit_triangle(r, triangles[0])) return vec3(0.1, 0.7, 0.41);

	vec3 unit_direction = unit_vector(r.GetDirection());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}


__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner,
	vec3 horizontal, vec3 vertical, vec3 origin, Triangle* triBuffer, int nrTris, int* dev_a, vec3* dev_v) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);
	vec3 colour = vec3(0, 0, dev_a[4]);
	fb[pixel_index] = colour;

	return;
	for (int w = 0; w < 1; w++) {
		Triangle tri({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 });
		const vec3 orig = r.GetOrigin();
		const vec3 dir = r.GetDirection();
		const vec3 move(0, 0, 0);
		vec3 v0 = move + tri.GetA();
		vec3 v1 = move + tri.GetB();
		vec3 v2 = move + tri.GetC();
		vec3 v0v1 = v1 - v0;
		vec3 v0v2 = v2 - v0;
		vec3 pvec = cross(dir, v0v2);
		float det = dot(v0v1, pvec);

		float invDet = 1 / det;

		vec3 tvec = orig - v0;

		float u = dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1) continue;

		vec3 qvec = cross(tvec, v0v1);
		float v = dot(dir, qvec) * invDet;
		if (v < 0 || u + v > 1) continue;

		float t = dot(v0v2, qvec) * invDet;
		if (t > 0) {
			colour = vec3(1, 1, 1);
			break;
		}

	}



	//Triangle tri({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 });
	//vec3 colour = vec3(0, 0, 0);
	////for (int k = 0; k < nrTris/2; k++) {
	//	if (hit_triangle(r, triBuffer[412])) {
	//		colour = vec3(1, 1, 1);
	//		//break;
	//	}
	//}
	//fb[pixel_index] = color(r, triBuffer, nrTris);
	fb[pixel_index] = colour;
}



int main()
{
	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);
	Mesh spyro = Mesh("Spyro/Spyro.obj");



	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	const Triangle  tris[1] = { Triangle({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 }) }; //!
	const int nrFaces = 1;
	size_t triangleBufferSize =   sizeof(Triangle); //!

	Triangle* dev_triangleBuffer = 0;

	checkCudaErrors(cudaMalloc((void**)&dev_triangleBuffer, triangleBufferSize));
	checkCudaErrors(cudaMemcpy(dev_triangleBuffer, tris, triangleBufferSize, cudaMemcpyHostToDevice));

	const int arraySize = 5;
	const int a[arraySize] = { 5, 4, 3, 2 ,1 };
	int* dev_a = 0;

	checkCudaErrors(cudaMalloc((void**)&dev_a, arraySize * sizeof(int)));
	checkCudaErrors(cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice));

	const vec3 veclist[3] = { vec3(1,2,3),vec3(4,5,6) ,vec3(7,8,9) };
	int vsize = 3;
	vec3* dev_v = 0;

	checkCudaErrors(cudaMalloc((void**)&dev_v, vsize * sizeof(vec3)));
	checkCudaErrors(cudaMemcpy(dev_v, veclist, vsize * sizeof(vec3), cudaMemcpyHostToDevice));



	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	render << <blocks, threads >> > (fb, nx, ny, vec3(-2.0, -1.0, -1.0),
		vec3(4.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 5.0),
		dev_triangleBuffer, nrFaces, dev_a, dev_v);


	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";
	try {
		// Output FB as Image
		std::cout << "P3\n" << nx << " " << ny << "\n255\n";
		for (int j = ny - 1; j >= 0; j--) {
			for (int i = 0; i < nx; i++) {
				size_t pixel_index = j * nx + i;
				int ir = int(255.99 * fb[pixel_index].r());
				int ig = int(255.99 * fb[pixel_index].g());
				int ib = int(255.99 * fb[pixel_index].b());
				std::cout << ir << " " << ig << " " << ib << "\n";
			}
		}
	}
	catch (int e) {
		std::cerr << "An exception occurred. Exception Nr. " << e << '\n';
	}
	checkCudaErrors(cudaFree(fb));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCudaErrors(cudaDeviceReset());


	return 0;


}

