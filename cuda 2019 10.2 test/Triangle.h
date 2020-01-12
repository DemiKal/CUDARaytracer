#pragma once
struct Triangle  {
private:
	vec3 a, b, c;
public:
	__device__ Triangle(const vec3& A, const vec3& B, const vec3& C) { a = A; b = B; c = C; }
	__device__ vec3 GetA() { return a; }
	__device__ vec3 GetB() { return b; }
	__device__ vec3 GetC() { return c; }

};

