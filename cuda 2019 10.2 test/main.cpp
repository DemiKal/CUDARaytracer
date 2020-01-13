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

texture<float4, 1, cudaReadModeElementType> texInput;

__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner,
	vec3 horizontal, vec3 vertical, vec3 origin, Triangle* triBuffer, int nrTris, int* dev_a, vec3* dev_v) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);

	//if ((i == 0) || (j == 0))
	//{
	//	float4 A = tex1Dfetch(texInput, 0);
	//	float4 B = tex1Dfetch(texInput, 1);
	//	float4 C = tex1Dfetch(texInput, 2);
	//
	//
	//	return;
	//}
	//vec3 colour = vec3(0, 0, 1);
	//fb[pixel_index] = colour;

	//return;
	//for (int w = 0; w < 1; w++) {
	//	Triangle tri({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 });
	//	const vec3 orig = r.GetOrigin();
	//	const vec3 dir = r.GetDirection();
	//	const vec3 move(0, 0, 0);
	//	vec3 v0 = move + tri.GetA();
	//	vec3 v1 = move + tri.GetB();
	//	vec3 v2 = move + tri.GetC();
	//	vec3 v0v1 = v1 - v0;
	//	vec3 v0v2 = v2 - v0;
	//	vec3 pvec = cross(dir, v0v2);
	//	float det = dot(v0v1, pvec);
	//
	//	float invDet = 1 / det;
	//
	//	vec3 tvec = orig - v0;
	//
	//	float u = dot(tvec, pvec) * invDet;
	//	if (u < 0 || u > 1) continue;
	//
	//	vec3 qvec = cross(tvec, v0v1);
	//	float v = dot(dir, qvec) * invDet;
	//	if (v < 0 || u + v > 1) continue;
	//
	//	float t = dot(v0v2, qvec) * invDet;
	//	if (t > 0) {
	//		colour = vec3(1, 1, 1);
	//		break;
	//	}
	//
	//}


	//Triangle tri({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 });
	vec3 colour = vec3(0, 0, 0);
	//if (i > 0 || j > 0) return;
	const vec3 orig = r.GetOrigin();
	const vec3 dir = r.GetDirection();
	const vec3 move(0, 0, 0);

	for (int k = 0; k < 150; k++) {

		float4 A = tex1Dfetch(texInput, 3*  k + 0);
		float4 B = tex1Dfetch(texInput, 3*  k + 1);
		float4 C = tex1Dfetch(texInput, 3*  k + 2);

 		vec3 v0 = move + vec3(A.x, A.y, A.z);
		vec3 v1 = move + vec3(B.x, B.y, B.z);
		vec3 v2 = move + vec3(C.x, C.y, C.z);
		vec3 v0v1 = v1 - v0;
		vec3 v0v2 = v2 - v0;
		vec3 pvec = cross(dir, v0v2);
		float det = dot(v0v1, pvec);

		float invDet = 1 / det;

		vec3 tvec = orig - v0;

		float u = dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1)  continue;  

		vec3 qvec = cross(tvec, v0v1);
		float v = dot(dir, qvec) * invDet;
		if (v < 0 || u + v > 1)   continue;  

		float t = dot(v0v2, qvec) * invDet;
		colour = vec3(1.f, 0.3f, .7f);
		break;
		//return t > 0;








		//vec3 Aa(A.x, A.y, A.z);
		//vec3 Bb(B.x, B.y, B.z);
		//vec3 Cc(C.x, C.y, C.z);
		//
		//Triangle tri(Aa, Bb, Cc);
		//
		//if (hit_triangle(r, tri)) {
		//	colour = vec3(1.f, 0.3f, .7f);
		//	break;
		//}
	}
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
	vec3 center = spyro.CalcCenter();
	float* dev_triangle_p;


	int number_of_triangles = spyro.m_faces.size();
	std::vector<float4> host_triangles;
	int tribufferSize = number_of_triangles * sizeof(Triangle);

	for (auto& f : spyro.m_faces) {
		//vec3 c = (f.GetA() + f.GetB() + f.GetC()) / 3.0f;
		//sum += c;
		vec3 A = f.GetA();
		vec3 B = f.GetB();
		vec3 C = f.GetC();

		host_triangles.emplace_back(make_float4(A.x(), A.y(), A.z(), 0));
		host_triangles.emplace_back(make_float4(B.x(), B.y(), B.z(), 0));
		host_triangles.emplace_back(make_float4(C.x(), C.y(), C.z(), 0));

	}


	texInput.addressMode[0] = cudaAddressModeBorder;
	texInput.addressMode[1] = cudaAddressModeBorder;
	texInput.filterMode = cudaFilterModePoint;
	texInput.normalized = false;
	float* dev_triangles = 0;
	size_t offset = 0;

	size_t coalescedSize = number_of_triangles * 3 * sizeof(float4);

	checkCudaErrors(cudaMalloc((void**)&dev_triangles, coalescedSize));

	checkCudaErrors(cudaMemcpy(dev_triangles, &host_triangles[0], coalescedSize, cudaMemcpyHostToDevice));


	const cudaChannelFormatDesc cd = cudaCreateChannelDesc<float4>();
	//channelDesc = cudaChannelFormatKindFloat;
	checkCudaErrors(cudaBindTexture(&offset, &texInput, dev_triangles, &cd, coalescedSize));

	// allocate memory for the triangle meshes on the GPU
	//cudaMalloc((void**)&dev_triangle_p, tribufferSize);
	//
	//// copy triangle data to GPU
	//cudaMemcpy(dev_triangle_p, &spyro.m_faces[0], tribufferSize, cudaMemcpyHostToDevice);
	//
	//// load triangle data into a CUDA texture
	////bindTriangles(dev_triangle_p, total_num_triangles);
	//
	//triangle_texture.normalized = false;                      // access with normalized texture coordinates
	//triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	//triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates
	//
	//size_t size = sizeof(Triangle) * number_of_triangles;
	//const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float3>();
	////cudaBindTexture(0, triangle_texture, dev_triangle_p, &channelDesc, size);



	//size_t size = sizeof(float4) * number_of_triangles * 3;
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	//cudaBindTexture(0, triangle_texture, dev_triangle_p, channelDesc, size);


	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	//const Triangle  tris[1] = { Triangle({ -4,-1, -3 }, { 2,-1, -3 }, { 0, 2,-3 }) }; //!
	const int nrFaces = spyro.m_faces.size();
	size_t triangleBufferSize = nrFaces * sizeof(Triangle); //!

	Triangle* dev_triangleBuffer = 0;

	checkCudaErrors(cudaMalloc((void**)&dev_triangleBuffer, triangleBufferSize));
	checkCudaErrors(cudaMemcpy(dev_triangleBuffer, &spyro.m_faces[0],
		triangleBufferSize, cudaMemcpyHostToDevice));

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

