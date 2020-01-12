#include "pch.h"
void Mesh::LoadMesh(const std::string& filename)
{

	const std::string inputfile = filename; // Clockwise initialized triangles
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string war, err;
	std::string directory = filename.substr(0, filename.find_last_of('/'));
	//bool ret;
	 bool ret =  LoadObj(
	 	&attrib,
	 	&shapes,
	 	&materials,
	 	&war,
	 	&err,
	 	inputfile.c_str(),
	 	directory.c_str());

	if (!err.empty()) // `err` may contain warning message.
	{
		std::cerr << err << std::endl;
	}

	if (!ret)
	{
		std::cerr << "error loading obj!" << std::endl;

		// throw    exception();
		exit(1);
	}



	//get starting idx of this buffer
	const int triangleStartIdx = 0;// mesh_vertices.size();

	// Loop over shapes
	//copy of vertices/faces for calc bounding box
	std::vector<vec3> mesh_vertices;
	std::vector<vec3> mesh_normals_preload;
	std::vector<vec2f> mesh_UVs_preload;

	for (int s = 0; s < shapes.size(); s++)
	{
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
		{
			size_t fv = shapes[s].mesh.num_face_vertices[f];
			std::vector<vec3> vertices;
			std::vector<vec3> normals;

			// Loop over vertices in the face.

			for (size_t v = 0; v < fv; v++)
			{
				// access to vertex

				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				//faces
				float vx = attrib.vertices[3  * idx.vertex_index + 0];
				float vy = attrib.vertices[3  * idx.vertex_index + 1];
				float vz = attrib.vertices[3  * idx.vertex_index + 2];

				//normals
				float nx = attrib.normals[3  * idx.normal_index + 0];
				float ny = attrib.normals[3  * idx.normal_index + 1];
				float nz = attrib.normals[3  * idx.normal_index + 2];

				//UVs
				float tx = attrib.texcoords[2  * idx.texcoord_index + 0];
				float ty = attrib.texcoords[2  * idx.texcoord_index + 1];
				vec3 vertex = vec3(vx, vy, vz);

				mesh_vertices.emplace_back(vertex);

				mesh_normals_preload.emplace_back(vec3(nx, ny, nz));
				mesh_UVs_preload.emplace_back(vec2f(tx, ty));
			}

			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}

	//calc bounding box
	float maxX = -1;
	float maxY = -1;
	float maxZ = -1;

	float minX = 999999;
	float minY = 999999;
	float minZ = 999999;

	//calc bounds
//for (int i = 0; i < mesh_vertices.size(); i++)
//{
//	vec3 v = mesh_vertices[i];
//
//	if (v.x > maxX) maxX = v.x;
//	if (v.x < minX) minX = v.x;
//
//	if (v.y > maxY) maxY = v.y;
//	if (v.y < minY) minY = v.y;
//
//	if (v.z > maxZ) maxZ = v.z;
//	if (v.z < minZ) minZ = v.z;
//}
	//aabb b(vec3{ minX, minY, minZ }, vec3{ maxX, maxY, maxZ });
	//mesh_bounds.emplace_back( b );

	//TODO: these indices should correspond to uv and normal?
	const size_t verticesEndIndex = mesh_vertices.size();
	//(unless using normal/uv mapping with texture

	//triangle data in contiguous array
	for (size_t i = triangleStartIdx; i < verticesEndIndex; i += 3)
	{
		const vec3 a = mesh_vertices[i + 0l];
		const vec3 b = mesh_vertices[i + 1l];
		const vec3 c = mesh_vertices[i + 2l];

		const Triangle t(a, b, c);
		m_faces.emplace_back(t);
		//const Triangle t_w(a + position, b + position, c + position);

		//mesh_triangles_local.emplace_back(t);
		//mesh_triangles_world.emplace_back(t_w);
//const float TminX = std::min(a.x, std::min(b.x, c.x));
			//const float TmaxX = std::max(a.x, std::max(b.x, c.x));
			//const float TminY = std::min(a.y, std::min(b.y, c.y));
			//const float TmaxY = std::max(a.y, std::max(b.y, c.y));
			//const float TminZ = std::min(a.z, std::min(b.z, c.z));
			//const float TmaxZ = std::max(a.z, std::max(b.z, c.z));
			//
			//aabb bbox(vec3{ TminX, TminY, TminZ }, vec3{ TmaxX, TmaxY, TmaxZ });
			//mesh_triangle_AABB.emplace_back(bbox);
	}

	//convert flat faces into struct so indexing is consistent
	const size_t preload_face_size = mesh_normals_preload.size();
	for (size_t i = 0; i < preload_face_size; i += 3l)
	{
		vec3 a = mesh_normals_preload[i];
		vec3 b = mesh_normals_preload[i + 1l];
		vec3 c = mesh_normals_preload[i + 2l];
		m_normals.emplace_back(Triangle(a, b, c));
	}

	const size_t preload_UV_size = mesh_UVs_preload.size();
	for (size_t i = 0; i < preload_UV_size; i += 3l)
	{
		vec2f a = mesh_UVs_preload[i];
		vec2f b = mesh_UVs_preload[i + 1l];
		vec2f c = mesh_UVs_preload[i + 2l];
		m_UVs.emplace_back(UV(a, b, c));
	}

	//const int triangleEndIndex = mesh_triangles_local.size();
	//Index idxBounds = Index(triangleStartIdx, triangleEndIndex, filepathObj);
	//mesh_triangle_index.emplace_back(idxBounds);


}