#pragma once


class Mesh
{
public:
	std::vector<Triangle> m_faces;
	std::vector<UV> m_UVs;
	std::vector<Triangle> m_normals;
	std::string name;

	vec3 CalcCenter() {
		vec3 sum(0, 0, 0);
		for (auto& f : m_faces) {
			vec3 c = (f.GetA() + f.GetB() + f.GetC()) / 3.0f;
			sum += c;
		}

		sum /= m_faces.size();
		return sum;
	}

	Mesh(const std::string& filename) {
		name = filename;
		LoadMesh(filename);
	}

	void LoadMesh(const std::string& filename);


};

