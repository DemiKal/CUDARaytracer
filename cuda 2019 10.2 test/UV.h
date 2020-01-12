#pragma once
class UV
{

public:

	vec2f A, B, C;

	//every vertex (3 per triangle) has a reference to a Vec2 with component UV
	UV(const vec2f& a, const vec2f& b, const vec2f& c) {
		A = a; B = b; C = c;
	}
};

