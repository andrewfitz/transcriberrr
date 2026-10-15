// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TranscriberrrUI",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "transcriberrr-ui", targets: ["TranscriberrrUI"]),
        .executable(name: "transcriberrr-native", targets: ["TranscriberrrNative"])
    ],
    targets: [
        .executableTarget(name: "TranscriberrrUI"),
        .executableTarget(name: "TranscriberrrNative")
    ]
)
