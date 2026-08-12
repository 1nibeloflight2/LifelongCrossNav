//
// Created by finn on 8/4/24.
//

#ifndef PATH__MATRIX_UTILS_H_
#define PATH__MATRIX_UTILS_H_
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
// Enums for data types and storage orders
enum class DataType { Int32, Int64, Bool, Float32, Float64, UInt8};
enum class StorageOrder { RowMajor, ColMajor };

// Enum-to-type mapping for data types
template <DataType> struct DataTypeToType;
template <> struct DataTypeToType<DataType::Int32> { using type = int32_t; };
template <> struct DataTypeToType<DataType::Int64> { using type = int64_t; };
template <> struct DataTypeToType<DataType::Float32> { using type = float; };
template <> struct DataTypeToType<DataType::Float64> { using type = double; };
template <> struct DataTypeToType<DataType::Bool>  { using type = bool; };
template <> struct DataTypeToType<DataType::UInt8>  { using type = uint8_t; };

// Enum-to-type mapping for storage orders
template <StorageOrder> struct StorageOrderToType;
template <> struct StorageOrderToType<StorageOrder::RowMajor> { static constexpr int value = Eigen::RowMajor; };
template <> struct StorageOrderToType<StorageOrder::ColMajor> { static constexpr int value = Eigen::ColMajor; };

// Helper to create the Eigen::Map type
template <DataType DT, StorageOrder SO>
struct MatrixTypeHelper {
    using T = typename DataTypeToType<DT>::type;
    static constexpr int Options = StorageOrderToType<SO>::value;
    using type = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Options>;
};

// Define all possible matrix types
using Int32RowMajor = Eigen::Map<const MatrixTypeHelper<DataType::Int32, StorageOrder::RowMajor>::type>;
using Int32ColMajor = Eigen::Map<const MatrixTypeHelper<DataType::Int32, StorageOrder::ColMajor>::type>;
using Int64RowMajor = Eigen::Map<const MatrixTypeHelper<DataType::Int64, StorageOrder::RowMajor>::type>;
using Int64ColMajor = Eigen::Map<const MatrixTypeHelper<DataType::Int64, StorageOrder::ColMajor>::type>;
using BoolRowMajor = Eigen::Map<const MatrixTypeHelper<DataType::Bool, StorageOrder::RowMajor>::type>;
using BoolColMajor = Eigen::Map<const MatrixTypeHelper<DataType::Bool, StorageOrder::ColMajor>::type>;
using Float32RowMajor = Eigen::Map<const MatrixTypeHelper<DataType::Float32, StorageOrder::RowMajor>::type>;
using Float32ColMajor = Eigen::Map<const MatrixTypeHelper<DataType::Float32, StorageOrder::ColMajor>::type>;
using Float64RowMajor = Eigen::Map<const MatrixTypeHelper<DataType::Float64, StorageOrder::RowMajor>::type>;
using Float64ColMajor = Eigen::Map<const MatrixTypeHelper<DataType::Float64, StorageOrder::ColMajor>::type>;
using UInt8RowMajor = Eigen::Map<const MatrixTypeHelper<DataType::UInt8, StorageOrder::RowMajor>::type>;
using UInt8ColMajor = Eigen::Map<const MatrixTypeHelper<DataType::UInt8, StorageOrder::ColMajor>::type>;

// Define the variant type that can hold any of these matrix types
using MatrixVariant = std::variant<
    Int32RowMajor, Int32ColMajor,
    Int64RowMajor, Int64ColMajor,
    BoolRowMajor, BoolColMajor,
    Float32RowMajor, Float32ColMajor,
    Float64RowMajor, Float64ColMajor,
    UInt8RowMajor, UInt8ColMajor
>;

// Helper to create the matrix
template <DataType DT, StorageOrder SO>
MatrixVariant create_matrix(const py::array& array) {
    using MatrixType = typename MatrixTypeHelper<DT, SO>::type;
    using T = typename DataTypeToType<DT>::type;

    return Eigen::Map<const MatrixType>(
        static_cast<const T*>(array.data()), array.shape(0), array.shape(1));
}

// Main function to get the matrix
MatrixVariant get_matrix(const py::array& array) {
    DataType dtype;
    if (array.dtype().kind() == 'i') {
        if (array.itemsize() == 4) {
            dtype = DataType::Int32;
        } else if (array.itemsize() == 8) {
            dtype = DataType::Int64;
        } else {
            throw std::runtime_error("Unsupported integer size");
        }
    } else if (array.dtype().kind() == 'f') {
        if (array.itemsize() == 4) {
            dtype = DataType::Float32;
        } else if (array.itemsize() == 8) {
            dtype = DataType::Float64;
        } else {
            throw std::runtime_error("Unsupported float size");
        }
    } else if (array.dtype().kind() == 'b') {
        dtype = DataType::Bool;
    } else if (array.dtype().kind() == 'u') {
        if (array.itemsize() == 1) {
            dtype = DataType::UInt8;
        } else {
            throw std::runtime_error("Unsupported unsigned integer size");
        }
    }
    else {
        throw std::runtime_error("Unsupported data type");
    }

    StorageOrder order = (array.flags() & py::array::f_style)
                         ? StorageOrder::ColMajor
                         : StorageOrder::RowMajor;

    switch (dtype) {
        case DataType::Int32:
            return (order == StorageOrder::ColMajor)
                   ? create_matrix<DataType::Int32, StorageOrder::ColMajor>(array)
                   : create_matrix<DataType::Int32, StorageOrder::RowMajor>(array);
        case DataType::Int64:
            return (order == StorageOrder::ColMajor)
                   ? create_matrix<DataType::Int64, StorageOrder::ColMajor>(array)
                   : create_matrix<DataType::Int64, StorageOrder::RowMajor>(array);
        case DataType::Bool:
            return (order == StorageOrder::ColMajor)
                   ? create_matrix<DataType::Bool, StorageOrder::ColMajor>(array)
                   : create_matrix<DataType::Bool, StorageOrder::RowMajor>(array);
        case DataType::Float32:
            return (order == StorageOrder::ColMajor)
                   ? create_matrix<DataType::Float32, StorageOrder::ColMajor>(array)
                   : create_matrix<DataType::Float32, StorageOrder::RowMajor>(array);
        case DataType::Float64:
            return (order == StorageOrder::ColMajor)
                   ? create_matrix<DataType::Float64, StorageOrder::ColMajor>(array)
                   : create_matrix<DataType::Float64, StorageOrder::RowMajor>(array);
        case DataType::UInt8:
            return (order == StorageOrder::ColMajor)
                   ? create_matrix<DataType::UInt8, StorageOrder::ColMajor>(array)
                   : create_matrix<DataType::UInt8, StorageOrder::RowMajor>(array);
        default:
            throw std::runtime_error("Unhandled data type");
    }
}
#endif //PATH__MATRIX_UTILS_H_
