// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "VTSCTuner",
  platforms: [
    .macOS(.v14),
  ],
  products: [
    .library(name: "VTSCTunerCore", targets: ["VTSCTunerCore"]),
    .executable(name: "VTSCTuner", targets: ["VTSCTunerApp"]),
  ],
  targets: [
    .target(name: "VTSCTunerCore"),
    .executableTarget(
      name: "VTSCTunerApp",
      dependencies: ["VTSCTunerCore"]
    ),
    .testTarget(
      name: "VTSCTunerCoreTests",
      dependencies: ["VTSCTunerCore"]
    ),
    .testTarget(
      name: "VTSCTunerAppTests",
      dependencies: ["VTSCTunerApp", "VTSCTunerCore"]
    ),
  ]
)
