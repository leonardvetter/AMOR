/*
 * U. Villa
 */

#include <cmath>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshFunction.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace dl = dolfin;

void numpy2MeshFunction3D(dl::Mesh & mesh,
						double h_x,
						double h_y,
						double h_z,
						double o_x,
						double o_y,
						double o_z,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i,j,k;
	auto dd = data.unchecked<3>();

	const int i_max = data.shape(0);
	const int j_max = data.shape(1);
	const int k_max = data.shape(2);

	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor( (p.x()-o_x)/h_x));
        j = int(std::floor( (p.y()-o_y)/h_y));
        k = int(std::floor( (p.z()-o_z)/h_z));

        if(i<0)
        	i=0;
        if(i >= i_max)
        	i = i_max-1;
        if(j<0)
        	j=0;
        if(j >= j_max)
        	j = j_max-1;
        if(k<0)
        	k=0;
        if(k >= k_max)
        	k = k_max-1;

        mfun[*cell] = dd(i,j,k);
	}
}


void numpy2MeshFunction2D(dl::Mesh & mesh,
						double h_x,
						double h_y,
						double o_x,
						double o_y,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i,j;

	auto dd = data.unchecked<2>();

	const int i_max = data.shape(0);
	const int j_max = data.shape(1);

	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor( (p.x()-o_x)/h_x));
        j = int(std::floor( (p.y()-o_y)/h_y));

        if(i<0)
        	i=0;
        if(i >= i_max)
        	i = i_max-1;
        if(j<0)
        	j=0;
        if(j >= j_max)
        	j = j_max-1;

        mfun[*cell] = dd(i,j);
	}
}

void numpy2MeshFunction1D(dl::Mesh & mesh,
						double h,
						double offset,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i;
	auto dd = data.unchecked<1>();

	const int i_max = data.shape(0);
	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor( (p.x() - offset)/h));

        if(i<0)
        	i=0;
        if(i >= i_max)
        	i = i_max-1;

        mfun[*cell] = dd(i);
	}
}

void numpy2MeshFunction(dl::Mesh & mesh,
						py::array_t<double> & h,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	auto hh = h.unchecked<1>();
	if(mesh.geometry().dim() == 3)
		numpy2MeshFunction3D(mesh, hh(0), hh(1), hh(2), 0., 0., 0., data, mfun);
	else if (mesh.geometry().dim() == 2)
		numpy2MeshFunction2D(mesh, hh(0), hh(1), 0., 0., data, mfun);
	else
		numpy2MeshFunction1D(mesh, hh(0), 0., data, mfun);
}

void numpy2MeshFunction_with_offsets(dl::Mesh & mesh,
						py::array_t<double> & h,
						py::array_t<double> & offsets,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	auto hh = h.unchecked<1>();
	auto oo = offsets.unchecked<1>();
	if(mesh.geometry().dim() == 3)
		numpy2MeshFunction3D(mesh, hh(0), hh(1), hh(2), oo(0), oo(1), oo(2), data, mfun);
	else if (mesh.geometry().dim() == 2)
		numpy2MeshFunction2D(mesh, hh(0), hh(1), oo(0), oo(1), data, mfun);
	else
		numpy2MeshFunction1D(mesh, hh(0), oo(0), data, mfun);
}
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <vector>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Constant.h>


namespace py = pybind11;

class NumpyScalarExpression3D : public dolfin::Expression
{
public:
	NumpyScalarExpression3D() :
    Expression(),
    data(),
    h_x(1.),
	h_y(1.),
	h_z(1.),
	off_x(0.),
	off_y(0.),
	off_z(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int j_m(std::floor( (x[1]-off_y) /h_y));
    int k_m(std::floor( (x[2]-off_z) /h_z));

    int i_p(std::ceil( (x[0]-off_x) /h_x));
    int j_p(std::ceil( (x[1]-off_y) /h_y));
    int k_p(std::ceil( (x[2]-off_z) /h_z));

    const int i_max = data.shape(0);
    const int j_max = data.shape(1);
    const int k_max = data.shape(2);


    if (i_p < 0 || i_m >= i_max || j_p < 0 || j_m >= j_max || k_m < 0 || k_p >= k_max)
    {
    	values[0] = 0.;
    	return;
    }

    auto dd = data.unchecked<3>();

    if(i_m < 0)
    	i_m = 0;
    if(i_p >= i_max)
    	i_p = i_max-1;
    if(j_m < 0)
    	j_m = 0;
    if(j_p >= j_max)
    	j_p = j_max-1;
    if(k_m < 0)
    	k_m = 0;
    if(k_p >= k_max)
    	k_p = k_max-1;

    const double alpha_x = (x[0] - off_x)/h_x - static_cast<double>(i_m);
    const double alpha_y = (x[1] - off_y)/h_y - static_cast<double>(j_m);
    const double alpha_z = (x[2] - off_z)/h_z - static_cast<double>(k_m);


    values[0] = (1. - alpha_x)*(1. - alpha_y)*(1. - alpha_z)*dd(i_m, j_m, k_m) +
    			(1. - alpha_x)*(1. - alpha_y)*      alpha_z *dd(i_m, j_m, k_p) +
				(1. - alpha_x)*      alpha_y *(1. - alpha_z)*dd(i_m, j_p, k_m) +
				(1. - alpha_x)*      alpha_y *      alpha_z *dd(i_m, j_p, k_p) +
				      alpha_x *(1. - alpha_y)*(1. - alpha_z)*dd(i_p, j_m, k_m) +
				      alpha_x *(1. - alpha_y)*      alpha_z *dd(i_p, j_m, k_p) +
				      alpha_x *      alpha_y *(1. - alpha_z)*dd(i_p, j_p, k_m) +
				      alpha_x *      alpha_y *      alpha_z *dd(i_p, j_p, k_p);

  }

  void setData(py::array_t<double>& d, double & hh_x, double & hh_y, double & hh_z)
  {
	  data = d;
	  h_x = hh_x;
	  h_y = hh_y;
	  h_z = hh_z;
  }
  void setOffset(double & offset_x, double & offset_y, double & offset_z)
  {
	  off_x = offset_x;
	  off_y = offset_y;
	  off_z = offset_z;
  }

private:
  py::array_t<double> data;
  double h_x;
  double h_y;
  double h_z;
  double off_x;
  double off_y;
  double off_z;

};

class NumpyVectorExpression3D : public dolfin::Expression
{
public:
	NumpyVectorExpression3D(int ncomp) :
    Expression(ncomp),
    ncomp(ncomp),
    data(),
    h_x(1.),
	h_y(1.),
	h_z(1.),
	off_x(0.),
	off_y(0.),
	off_z(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int j_m(std::floor( (x[1]-off_y) /h_y));
    int k_m(std::floor( (x[2]-off_z) /h_z));

    int i_p(std::ceil( (x[0]-off_x) /h_x));
    int j_p(std::ceil( (x[1]-off_y) /h_y));
    int k_p(std::ceil( (x[2]-off_z) /h_z));

    const int i_max = data.shape(0);
    const int j_max = data.shape(1);
    const int k_max = data.shape(2);


    if (i_p < 0 || i_m >= i_max || j_p < 0 || j_m >= j_max || k_m < 0 || k_p >= k_max)
    {
    	values[0] = 0.;
    	return;
    }

    auto dd = data.unchecked<4>();

    if(i_m < 0)
    	i_m = 0;
    if(i_p >= i_max)
    	i_p = i_max-1;
    if(j_m < 0)
    	j_m = 0;
    if(j_p >= j_max)
    	j_p = j_max-1;
    if(k_m < 0)
    	k_m = 0;
    if(k_p >= k_max)
    	k_p = k_max-1;

    const double alpha_x = (x[0] - off_x)/h_x - static_cast<double>(i_m);
    const double alpha_y = (x[1] - off_y)/h_y - static_cast<double>(j_m);
    const double alpha_z = (x[2] - off_z)/h_z - static_cast<double>(k_m);

    for( int icomp(0); icomp < ncomp; ++icomp)
    values[icomp] = (1. - alpha_x)*(1. - alpha_y)*(1. - alpha_z)*dd(i_m, j_m, k_m, icomp) +
    			(1. - alpha_x)*(1. - alpha_y)*      alpha_z *dd(i_m, j_m, k_p, icomp) +
				(1. - alpha_x)*      alpha_y *(1. - alpha_z)*dd(i_m, j_p, k_m, icomp) +
				(1. - alpha_x)*      alpha_y *      alpha_z *dd(i_m, j_p, k_p, icomp) +
				      alpha_x *(1. - alpha_y)*(1. - alpha_z)*dd(i_p, j_m, k_m, icomp) +
				      alpha_x *(1. - alpha_y)*      alpha_z *dd(i_p, j_m, k_p, icomp) +
				      alpha_x *      alpha_y *(1. - alpha_z)*dd(i_p, j_p, k_m, icomp) +
				      alpha_x *      alpha_y *      alpha_z *dd(i_p, j_p, k_p, icomp);

  }

  void setData(py::array_t<double>& d, double & hh_x, double & hh_y, double & hh_z)
  {
	  data = d;
	  h_x = hh_x;
	  h_y = hh_y;
	  h_z = hh_z;
  }
  void setOffset(double & offset_x, double & offset_y, double & offset_z)
  {
	  off_x = offset_x;
	  off_y = offset_y;
	  off_z = offset_z;
  }

private:
  int ncomp;
  py::array_t<double> data;
  double h_x;
  double h_y;
  double h_z;
  double off_x;
  double off_y;
  double off_z;

};

class NumpyScalarExpression2D : public dolfin::Expression
{
public:
	NumpyScalarExpression2D() :
    Expression(),
    data(),
    h_x(1.),
	h_y(1.),
	off_x(0.),
	off_y(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int i_p(std::ceil( (x[0]-off_x) /h_x));

    int j_m(std::floor( (x[1]-off_y) /h_y));
    int j_p(std::ceil( (x[1]-off_y) /h_y));

    const int i_max = data.shape(0);
    const int j_max = data.shape(1);

    auto dd = data.unchecked<2>();

    if(i_p < 0 || i_m >= i_max || j_p < 0 || j_m >= j_max)
    {
    	values[0] = 0.;
    	return;
    }

    if(i_m < 0)
    	i_m = 0;
    if(i_p >= i_max)
    	i_p = i_max-1;
    if(j_m < 0)
    	j_m = 0;
    if(j_p >= j_max)
    	j_p = j_max-1;

    const double alpha_x = (x[0] - off_x)/h_x - static_cast<double>(i_m);
    const double alpha_y = (x[1] - off_y)/h_y - static_cast<double>(j_m);

    values[0] = (1. - alpha_x)*(1. - alpha_y)*dd(i_m, j_m) +
    		    (1. - alpha_x)*      alpha_y *dd(i_m, j_p) +
				      alpha_x *(1. - alpha_y)*dd(i_p, j_m) +
				      alpha_x *      alpha_y *dd(i_p, j_p);
  }

  void setData(py::array_t<double>& d, double & hh_x, double & hh_y)
  {
	  data = d;
	  h_x = hh_x;
	  h_y = hh_y;
  }
  void setOffset(double & offset_x, double & offset_y)
  {
	  off_x = offset_x;
	  off_y = offset_y;
  }

private:
  py::array_t<double> data;
  double h_x;
  double h_y;
  double off_x;
  double off_y;

};

class NumpyVectorExpression2D : public dolfin::Expression
{
public:
	NumpyVectorExpression2D(int ncomp) :
    Expression(ncomp),
    ncomp(ncomp),
    data(),
    h_x(1.),
	h_y(1.),
	off_x(0.),
	off_y(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int i_p(std::ceil( (x[0]-off_x) /h_x));

    int j_m(std::floor( (x[1]-off_y) /h_y));
    int j_p(std::ceil( (x[1]-off_y) /h_y));

    const int i_max = data.shape(0);
    const int j_max = data.shape(1);

    auto dd = data.unchecked<3>();

    if(i_p < 0 || i_m >= i_max || j_p < 0 || j_m >= j_max)
    {
    	values[0] = 0.;
    	return;
    }

    if(i_m < 0)
    	i_m = 0;
    if(i_p >= i_max)
    	i_p = i_max-1;
    if(j_m < 0)
    	j_m = 0;
    if(j_p >= j_max)
    	j_p = j_max-1;

    const double alpha_x = (x[0] - off_x)/h_x - static_cast<double>(i_m);
    const double alpha_y = (x[1] - off_y)/h_y - static_cast<double>(j_m);

    for(int icomp(0); icomp < ncomp; ++icomp)
    values[icomp] = (1. - alpha_x)*(1. - alpha_y)*dd(i_m, j_m, icomp) +
    		    (1. - alpha_x)*      alpha_y *dd(i_m, j_p, icomp) +
				      alpha_x *(1. - alpha_y)*dd(i_p, j_m, icomp) +
				      alpha_x *      alpha_y *dd(i_p, j_p, icomp);
  }

  void setData(py::array_t<double>& d, double & hh_x, double & hh_y)
  {
	  data = d;
	  h_x = hh_x;
	  h_y = hh_y;
  }
  void setOffset(double & offset_x, double & offset_y)
  {
	  off_x = offset_x;
	  off_y = offset_y;
  }

private:
  int ncomp;
  py::array_t<double> data;
  double h_x;
  double h_y;
  double off_x;
  double off_y;

};

class NumpyScalarExpression1D : public dolfin::Expression
{
public:
	NumpyScalarExpression1D() :
    Expression(),
    data(),
    h_x(1.),
	off_x(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int i_p(std::ceil( (x[0]-off_x) /h_x));

    const int i_max = data.shape(0);
	auto dd = data.unchecked<1>();

    if(i_p < 0 || i_m > i_max)
    {
    	values[0] = 0.;
    	return;
    }

    if(i_m < 0)
    	i_m = 0;

    if(i_p > i_max)
    	i_p = i_max-1;

    const double alpha = (x[0] - off_x)/h_x - static_cast<double>(i_m);

    values[0] = (1. - alpha)*dd(i_m) + alpha*dd(i_p);
  }

  void setData(py::array_t<double>& d, double & hh_x)
  {
	  data = d;
	  h_x = hh_x;
  }
  void setOffset(double & offset_x)
  {
	  off_x = offset_x;
  }

private:
  py::array_t<double> data;
  double h_x;
  double off_x;

};

class NumpyVectorExpression1D : public dolfin::Expression
{
public:
	NumpyVectorExpression1D(int ncomp) :
    Expression(ncomp),
    ncomp(ncomp),
    data(),
    h_x(1.),
	off_x(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int i_p(std::ceil( (x[0]-off_x) /h_x));

    const int i_max = data.shape(0);
	auto dd = data.unchecked<2>();

    if(i_p < 0 || i_m > i_max)
    {
    	values[0] = 0.;
    	return;
    }

    if(i_m < 0)
    	i_m = 0;

    if(i_p > i_max)
    	i_p = i_max-1;

    const double alpha = (x[0] - off_x)/h_x - static_cast<double>(i_m);

    for(int icomp(0); icomp < ncomp; ++icomp)
    values[icomp] = (1. - alpha)*dd(i_m, icomp) + alpha*dd(i_p, icomp);
  }

  void setData(py::array_t<double>& d, double & hh_x)
  {
	  data = d;
	  h_x = hh_x;
  }
  void setOffset(double & offset_x)
  {
	  off_x = offset_x;
  }

private:
  int ncomp;
  py::array_t<double> data;
  double h_x;
  double off_x;

};PYBIND11_MODULE(dolfin_cpp_module_8f1a51deef6610859f699928d4e9f47b, m)
{
	m.def("numpy2MeshFunction",  &numpy2MeshFunction,
		  "Convert an ND numpy array to MeshFunction",
		  py::arg("mesh"),
		  py::arg("h"),
		  py::arg("data"),
		  py::arg("mfun"));

	m.def("numpy2MeshFunction",  &numpy2MeshFunction_with_offsets,
		  "Convert an ND numpy array to MeshFunction",
		  py::arg("mesh"),
		  py::arg("h"),
		  py::arg("offsets"),
		  py::arg("data"),
		  py::arg("mfun"));

    py::class_<NumpyScalarExpression3D, std::shared_ptr<NumpyScalarExpression3D>, dolfin::Expression>
    (m, "NumpyScalarExpression3D")
    .def(py::init<>(),"Interpolate a 3D numpy array on the mesh - trilinear interpolation")
    .def("setData", &NumpyScalarExpression3D::setData, py::arg("data"), py::arg("h_x"), py::arg("h_y"), py::arg("h_z"))
	.def("setOffset", &NumpyScalarExpression3D::setOffset, py::arg("offset_x"), py::arg("offset_y"), py::arg("offset_z"));

	py::class_<NumpyVectorExpression3D, std::shared_ptr<NumpyVectorExpression3D>, dolfin::Expression>
    (m, "NumpyVectorExpression3D")
    .def(py::init<int>(),"Interpolate a 4D numpy array on the mesh (the component of the field is the last dimension of the array) - trilinear interpolation")
    .def("setData", &NumpyVectorExpression3D::setData, py::arg("data"), py::arg("h_x"), py::arg("h_y"), py::arg("h_z"))
	.def("setOffset", &NumpyVectorExpression3D::setOffset, py::arg("offset_x"), py::arg("offset_y"), py::arg("offset_z"));

    py::class_<NumpyScalarExpression2D, std::shared_ptr<NumpyScalarExpression2D>, dolfin::Expression>
    (m, "NumpyScalarExpression2D", "Interpolate a 2D numpy array on the mesh - bilinear interpolation")
    .def(py::init<>())
    .def("setData", &NumpyScalarExpression2D::setData, py::arg("data"), py::arg("h_x"), py::arg("h_y"))
	.def("setOffset", &NumpyScalarExpression2D::setOffset, py::arg("offset_x"), py::arg("offset_y"));

	py::class_<NumpyVectorExpression2D, std::shared_ptr<NumpyVectorExpression2D>, dolfin::Expression>
    (m, "NumpyVectorExpression2D")
    .def(py::init<int>(),"Interpolate a 3D numpy array on the mesh (the component of the field is the last dimension of the array) - bilinear interpolation")
    .def("setData", &NumpyVectorExpression2D::setData, py::arg("data"), py::arg("h_x"), py::arg("h_y"))
	.def("setOffset", &NumpyVectorExpression2D::setOffset, py::arg("offset_x"), py::arg("offset_y"));

    py::class_<NumpyScalarExpression1D, std::shared_ptr<NumpyScalarExpression1D>, dolfin::Expression>
    (m, "NumpyScalarExpression1D", "Interpolate a 1D numpy array on the mesh - linear interpolation")
    .def(py::init<>())
    .def("setData", &NumpyScalarExpression1D::setData, py::arg("data"), py::arg("h_x"))
	.def("setOffset", &NumpyScalarExpression1D::setOffset, py::arg("offset_x"));

	py::class_<NumpyVectorExpression1D, std::shared_ptr<NumpyVectorExpression1D>, dolfin::Expression>
    (m, "NumpyVectorExpression1D")
    .def(py::init<int>(),"Interpolate a 2D numpy array on the mesh (the component of the field is the last dimension of the array) - linear interpolation")
    .def("setData", &NumpyVectorExpression1D::setData, py::arg("data"), py::arg("h_x"))
	.def("setOffset", &NumpyVectorExpression1D::setOffset, py::arg("offset_x"));
}
