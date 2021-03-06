# Description: Model Server

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "model_platform_types",
    hdrs = ["model_platform_types.h"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "platform_config_util",
    srcs = ["platform_config_util.cc"],
    hdrs = ["platform_config_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/config:platform_config_proto",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter_proto",
    ],
)

cc_library(
    name = "server_core",
    srcs = ["server_core.cc"],
    hdrs = ["server_core.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":model_platform_types",
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/config:logging_config_proto",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/config:platform_config_proto",
        "//tensorflow_serving/core:aspired_versions_manager",
        "//tensorflow_serving/core:dynamic_source_router",
        "//tensorflow_serving/core:load_servables_fast",
        "//tensorflow_serving/core:servable_state_monitor",
        "//tensorflow_serving/core:server_request_logger",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter",
        "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter",
        "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter_proto",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:optional",
        "//tensorflow_serving/util:unique_ptr_with_deps",
        "@org_tensorflow//tensorflow/core:lib",
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

cc_test(
    name = "server_core_test",
    size = "medium",
    srcs = ["server_core_test.cc"],
    deps = [
        ":model_platform_types",
        ":server_core",
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/core/test_util:availability_test_util",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter_proto",
        "//tensorflow_serving/core/test_util:fake_log_collector",
        "//tensorflow_serving/core/test_util:mock_request_logger",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers/test_util:server_core_test_util",
        "//tensorflow_serving/model_servers/test_util:storage_path_error_injecting_source_adapter",
        "//tensorflow_serving/model_servers/test_util:storage_path_error_injecting_source_adapter_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

SUPPORTED_TENSORFLOW_OPS = [
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]

TENSORFLOW_DEPS = [
    "@org_tensorflow//tensorflow/core:tensorflow",
    "//tensorflow_serving/servables/tensorflow:classification_service",
    "//tensorflow_serving/servables/tensorflow:get_model_metadata_impl",
    "//tensorflow_serving/servables/tensorflow:multi_inference",
    "//tensorflow_serving/servables/tensorflow:regression_service",
    "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter",
    "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter",
    "//tensorflow_serving/servables/tensorflow:predict_impl",
]

cc_binary(
    name = "tensorflow_model_server",
    srcs = [
        "main.cc",
    ],
    visibility = ["//tensorflow_serving:internal"],
    deps = [
        ":model_platform_types",
        ":platform_config_util",
        ":server_core",
        "@protobuf_archive//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform/cloud:gcs_file_system",
        "@org_tensorflow//tensorflow/core/platform/hadoop:hadoop_file_system",
        "//tensorflow_serving/apis:prediction_service_proto",
	"//tensorflow_serving/apis:configuration_service_proto",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "@grpc//:grpc++_unsecure",
    ] + TENSORFLOW_DEPS + SUPPORTED_TENSORFLOW_OPS,
)

py_test(
    name = "tensorflow_model_server_test",
    size = "medium",
    srcs = ["tensorflow_model_server_test.py"],
    data = [
        ":tensorflow_model_server",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_model_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:batching_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:good_model_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/assets/foo.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/saved_model.pb",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/variables/variables.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/variables/variables.index",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "tensorflow_model_server_test_client",
    srcs = ["tensorflow_model_server_test_client.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_deb", "pkg_tar")

pkg_tar(
    name = "tensorflow_model_server_tar",
    files = [
        ":tensorflow_model_server",
    ],
    package_dir = "/usr/bin",
)

# Build with '-c opt --copt=-mavx --copt=-msse4.2'
pkg_deb(
    name = "tensorflow_model_server_deb",
    data = ":tensorflow_model_server_tar",
    description = "TensorFlow Serving ModelServer",
    homepage = "https://github.com/tensorflow/serving",
    maintainer = "TensorFlow Serving team",
    package = "tensorflow-model-server",
    version = "undefined",  # Set when releasing a new version of TensorFlow Serving (e.g. 1.0.0).
)

# Build with '-c opt'
pkg_deb(
    name = "tensorflow_model_server_universal_deb",
    data = ":tensorflow_model_server_tar",
    description = "TensorFlow Serving ModelServer",
    homepage = "https://github.com/tensorflow/serving",
    maintainer = "TensorFlow Serving team",
    package = "tensorflow-model-server-universal",
    version = "undefined",  # Set when releasing a new version of TensorFlow Serving (e.g. 1.0.0).
)
