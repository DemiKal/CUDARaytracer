#include "pch.h"
__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
	vec3 oc = r.GetOrigin() - center;
	float a = dot(r.GetDirection(), r.GetDirection());
	float b = 2.0f * dot(oc, r.GetDirection());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	return (discriminant > 0.0f);
}