#pragma once
class ray
{
public:
	__device__ ray() {}
	__device__ ray(const vec3& orig, const vec3& dir) { O = orig; D = dir; }
	__device__ vec3 GetOrigin() const { return O; }
	__device__ vec3 GetDirection() const { return D; }
	__device__ vec3 point_at_parameter(float t) const { return O + t * D; }

private:
	vec3 O;
	vec3 D;
};
