#include "BasicScene.h"
#include <read_triangle_mesh.h>
#include <utility>
#include "ObjLoader.h"
#include "IglMeshLoader.h"
#include "igl/read_triangle_mesh.cpp"
#include "igl/edge_flaps.h"
#include "AutoMorphingModel.h"
#include <igl/circulation.h>
#include <igl/collapse_edge.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/per_vertex_normals.h>
#include <igl/parallel_for.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <iostream>
#include <set>

using namespace Eigen;
using namespace igl;
using namespace std;

using namespace cg3d;

// our data structures
Eigen::VectorXi EMAP;
Eigen::MatrixXi F, OF, E, EF, EI;
Eigen::VectorXi EQ;
Eigen::MatrixXd V, OV, C;
int num_collapsed;
Eigen::MatrixXd N, T, points, edges, colors;
igl::min_heap< std::tuple<double, int, int> > Q;

Eigen::Matrix4d calculateK(auto v1, auto v2, auto v3)
{
	// extract each vertex coordinates
	double x1 = v1(0); double y1 = v1(1); double z1 = v1(2);
	double x2 = v2(0); double y2 = v2(1); double z2 = v2(2);
	double x3 = v3(0); double y3 = v3(1); double z3 = v3(2);

	// calculates for a, b, c, d
	double a1 = x2 - x1; double b1 = y2 - y1; double c1 = z2 - z1;
	double a2 = x3 - x1; double b2 = y3 - y1; double c2 = z3 - z1;

	double b1c2 = b1 * c2;
	double b2c1 = b2 * c1;
	double a = b1c2 - b2c1;
	double a2c1 = a2 * c1;
	double a1c2 = a1 * c2;
	double b = a2c1 - a1c2;
	double a1b2 = a1 * b2;
	double b1a2 = b1 * a2;
	double c = a1b2 - b1a2;
	double d = (-a * x1 - b * y1 - c * z1);

	// normalized for finding a,b,c that maintain a^2 + b^2 + c^2 = 1
	double vectorSize = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
	a = a / vectorSize;
	b = b / vectorSize;
	c = c / vectorSize;
	d = d / vectorSize;

	// define 4*4 matrix and put the right values into it according to the formula
	Eigen::Matrix4d k;
	k << pow(a, 2), a* b, a* c, a* d,
		a* b, pow(b, 2), b* c, b* d,
		a* c, b* c, pow(c, 2), c* d,
		a* d, b* d, c* d, pow(d, 2);
	return k;
}

void lowest_cost_edge_and_midpoint(
	const int e,
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXi& E,
	const Eigen::VectorXi& /*EMAP*/,
	const Eigen::MatrixXi& /*EF*/,
	const Eigen::MatrixXi& /*EI*/,
	double& cost,
	Eigen::RowVectorXd& p)
{
	// extract the vertex coordinates from the edge e
	int v1 = E(e, 0);
	int v2 = E(e, 1);

	vector<int>* planes = new vector<int>();
	for (int j = 0; j < F.size() / 3; j++)
	{
		//find all connected faces to the vertex v1 and save the index of the faces
		if (F(j, 0) == v1 || F(j, 1) == v1 || F(j, 2) == v1)
			planes->push_back(j);
	}

	// define a 4 * 4 matrix of floats
	Eigen::Matrix4d K1;
	K1.setZero();

	//for every face find plane forumla: A B C D and sum it
	for (int j = 0; j < planes->size(); j++) {
		// take the face with that index
		auto face = F.row(planes->at(j));
		auto vertex1 = V.row(face(0));
		auto vertex2 = V.row(face(1));
		auto vertex3 = V.row(face(2));
		if ((vertex1 != vertex2) || (vertex1 != vertex3) || (vertex2 != vertex3)) {
			K1 += calculateK(vertex1, vertex2, vertex3);
		}
	}

	// define the v matrix
	Matrix<double, 4, 1> vertex1 = { double(V(v1, 0)), double(V(v1, 1)), double(V(v1, 2)), 1 };
	// calculate the cost(deltaV) of the vertex v1
	auto tmp1 = vertex1.transpose() * K1;
	double costV1 = tmp1 * vertex1;
	//====================================================================================v2
	// reset planes vector
	planes = new vector<int>();
	for (int j = 0; j < F.size() / 3; j++)
	{
		//find all connected faces to the vertex v2 and save the index of the faces
		if (F(j, 0) == v2 || F(j, 1) == v2 || F(j, 2) == v2)
			planes->push_back(j);
	}
	// define a 4 * 4 matrix of floats
	Eigen::Matrix4d K2;
	K2.setZero();

	//for every face find plane forumla: A B C D and sum it
	for (int j = 0; j < planes->size(); j++) {
		// take the face with that index
		auto face = F.row(planes->at(j));
		auto vertex1 = V.row(face(0));
		auto vertex2 = V.row(face(1));
		auto vertex3 = V.row(face(2));
		if ((vertex1 != vertex2) || (vertex1 != vertex3) || (vertex2 != vertex3)) {
			K2 += calculateK(vertex1, vertex2, vertex3);
		}
	}
	// define the v matrix
	Matrix<double, 4, 1> vertex2 = { double(V(v2, 0)), double(V(v2, 1)), double(V(v2, 2)), 1 };
	// calculate the cost(deltaV) of the vertex v2
	auto tmp2 = vertex2.transpose() * K2;
	double costV2 = tmp2 * vertex2;

	Eigen::Matrix4d Kt = K1 + K2;

	//midpoint
	auto tmp3 = ((vertex1 + vertex2) / 2).transpose() * Kt;
	double costT = tmp3 * ((vertex1 + vertex2) / 2);

	//cout << "cost1: " << costV1 << ", cost2: " << costV2 << ", costT: " << costT << endl;

	cost = costV1;
	p = V.row(E(e, 0));

	if (costV2 > cost) {
		cost = costV2;
		p = V.row(E(e, 1));
	}
	if (costT > cost) {
		cost = costT;
		p = 0.5 * (V.row(E(e, 0)) + V.row(E(e, 1)));
	}
}

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
	camera = Camera::Create("camera", fov, float(width) / height, near, far);

	AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
	auto daylight{ std::make_shared<Material>("daylight", "shaders/cubemapShader") };
	daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
	auto background{ Model::Create("background", Mesh::Cube(), daylight) };
	AddChild(background);
	background->Scale(120, Axis::XYZ);
	background->SetPickable(false);
	background->SetStatic();

	auto program = std::make_shared<Program>("shaders/basicShader");
	auto material{ std::make_shared<Material>("material", program) }; // empty material
	//    SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());

	material->AddTexture(0, "textures/box0.bmp", 2);
	auto sphereMesh{ IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj") };
	//auto cylMesh{ IglLoader::MeshFromFiles("cyl_igl","data/camel_b.obj") };
	//auto cubeMesh{ IglLoader::MeshFromFiles("cube_igl","data/cube.off") };

	sphere1 = Model::Create("sphere", sphereMesh, material);
	//cyl = Model::Create("cyl", cylMesh, material);
	//cube = Model::Create("cube", cubeMesh, material);

	// our code
	auto morphFunc = [](Model* model, cg3d::Visitor* visitor) {
		//return (model->GetMeshList())[0]->data.size() - 1;
		return model->meshIndex;
	};
	//auto autoCube = AutoMorphingModel::Create(*cube, morphFunc);
	auto autoSphere = AutoMorphingModel::Create(*sphere1, morphFunc);

	autoSphere->Scale(5);
	autoSphere->showWireframe = true;
	//autoSphere->Translate({ -3,0,0 });
	//cyl->Translate({ 3,0,0 });
	//cyl->Scale(0.12f);
	//cyl->showWireframe = true;
	//autoCube->showWireframe = true;
	//autoSphere->showWireframe = true;
	camera->Translate(20, Axis::Z);
	//root->AddChild(cyl);
	root->AddChild(autoSphere);
	//root->AddChild(autoCube);

	auto mesh = autoSphere->GetMeshList();

	// Function to reset original mesh and data structures
	OV = mesh[0]->data[0].vertices;
	OF = mesh[0]->data[0].faces;

	// Function to reset original mesh and data structures
	const auto& reset = [&]()
	{
		F = OF;
		V = OV;
		edge_flaps(F, E, EMAP, EF, EI);
		C.resize(E.rows(), V.cols());
		VectorXd costs(E.rows());
		// Q.clear();
		Q = {};
		EQ = Eigen::VectorXi::Zero(E.rows());
		int vCount = 0;
		{
			Eigen::VectorXd costs(E.rows());
			igl::parallel_for(E.rows(), [&](const int e)
				{
					double cost = e;
					RowVectorXd p(1, 3);
					per_vertex_normals(V, F, N);
					lowest_cost_edge_and_midpoint(e, V, F, E, EMAP, EF, EI, cost, p);
					printf("calculating initial costs: %d/%d\n", V.size(), vCount);
					vCount++;
					C.row(e) = p;
					costs(e) = cost;
				}, 10000);
			for (int e = 0; e < E.rows(); e++)
			{
				Q.emplace(costs(e), e, 0);
			}
		}

		num_collapsed = 0;
		igl::per_vertex_normals(V, F, N);
		T = Eigen::MatrixXd::Zero(V.rows(), 2);
	};

	reset();
	//std::cout << "vertices: " << V << std::endl;
	//std::cout << "faces: " << F << std::endl;
	//std::cout << "edges: " << E << std::endl;
	//std::cout << "edges to faces: \n" << EF.transpose() << std::endl;
	//std::cout << "faces to edges: \n " << EMAP.transpose() << std::endl;
	//std::cout << "edges indices: \n" << EI.transpose() << std::endl;
}

void BasicScene::Update(const Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model)
{
	Scene::Update(program, proj, view, model);
	program.SetUniform4f("lightColor", 1.0f, 1.0f, 1.0f, 0.5f);
	program.SetUniform4f("Kai", 1.0f, 1.0f, 1.0f, 1.0f);
	//cube->Rotate(0.01f, Axis::All);
}

void BasicScene::Simplification(int facesNum)
{
	if (pickedModel)
	{
		bool something_collapsed = false;
		for (int j = 0; j < facesNum; j++)
		{
			int edgeIndex = get<1>(Q.top());
			double cost;
			Eigen::RowVectorXd p;
			lowest_cost_edge_and_midpoint(edgeIndex, V, F, E, EMAP, EF, EI, cost, p);
			per_vertex_normals(V, F, N);
			if (!collapse_edge(lowest_cost_edge_and_midpoint, V, F, E, EMAP, EF, EI, Q, EQ, C))
			{
				printf("can't collapse edge\n");
				break;
			}
			cout << "edge " << edgeIndex << " ,cost = " << cost;
			printf(", new v position (%f,%f,%f)\n", p(0), p(1), p(2));
			something_collapsed = true;
			num_collapsed++;
			printf("total edges collapsed: %d\n", num_collapsed);
			//std::cout << "edges: " << E.size() / 2 << std::endl;
			//std::cout << "Q: " << Q.size() << std::endl;
			//printf("Q.top - cost: %f, edgeIndex: %d\n", get<0>(Q.top()), get<1>(Q.top()));
		}

		if (something_collapsed)
		{
			igl::per_vertex_normals(V, F, N);
			T = Eigen::MatrixXd::Zero(V.rows(), 2);
			auto mesh = pickedModel->GetMeshList();
			mesh[0]->data.push_back({ V, F, N, T });
			pickedModel->SetMeshList(mesh);
			pickedModel->meshIndex = mesh[0]->data.size() - 1;
		}
	}
	else
	{
		printf("no picked model\n");
	}
}

void BasicScene::KeyCallback(Viewport* viewport, int x, int y, int key, int scancode, int action, int mods)
{
	auto system = camera->GetRotation().transpose();

	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) // NOLINT(hicpp-multiway-paths-covered)
		{
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			break;
		case GLFW_KEY_UP:
			if (pickedModel) {
				int newIndex = pickedModel->meshIndex + 1;
				int maxIndex = pickedModel->GetMeshList()[0]->data.size() - 1;
				if (newIndex <= maxIndex)
					pickedModel->meshIndex = newIndex;
				printf("%d\n", pickedModel->meshIndex);

			}
			else {
				camera->RotateInSystem(system, 0.1f, Axis::X);
			}
			break;
		case GLFW_KEY_DOWN:
			if (pickedModel) {
				int newIndex = pickedModel->meshIndex - 1;
				int minIndex = 0;
				if (newIndex >= minIndex)
					pickedModel->meshIndex = newIndex;
				printf("%d\n", pickedModel->meshIndex);
			}
			else {
				camera->RotateInSystem(system, -0.1f, Axis::X);
			}
			break;
		case GLFW_KEY_LEFT:
			camera->RotateInSystem(system, 0.1f, Axis::Y);
			break;
		case GLFW_KEY_RIGHT:
			camera->RotateInSystem(system, -0.1f, Axis::Y);
			break;
		case GLFW_KEY_W:
			camera->TranslateInSystem(system, { 0, 0.05f, 0 });
			break;
		case GLFW_KEY_S:
			camera->TranslateInSystem(system, { 0, -0.05f, 0 });
			break;
		case GLFW_KEY_A:
			camera->TranslateInSystem(system, { -0.05f, 0, 0 });
			break;
		case GLFW_KEY_D:
			camera->TranslateInSystem(system, { 0.05f, 0, 0 });
			break;
		case GLFW_KEY_B:
			camera->TranslateInSystem(system, { 0, 0, 0.05f });
			break;
		case GLFW_KEY_F:
			camera->TranslateInSystem(system, { 0, 0, -0.05f });
			break;
		case GLFW_KEY_SPACE:
			//  reduce 10 % of the edges in the chosen model from the most simplified mesh
			Simplification(std::ceil(0.1 * (E.size() / 2)));
			break;
		case GLFW_KEY_V:
			Simplification(1);
			break;
		}
	}
}
