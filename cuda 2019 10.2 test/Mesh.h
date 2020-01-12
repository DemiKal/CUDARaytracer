#pragma once


class Mesh
{
public:
	std::vector<Triangle> m_faces;
	std::vector<UV> m_UVs;
	std::vector<Triangle> m_normals;
	std::string name;


	Mesh(const std::string& filename) {
		name = filename;
		LoadMesh(filename);
	}

	void LoadMesh(const std::string& filename);


};

