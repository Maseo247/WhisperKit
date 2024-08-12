import Foundation
import WhisperKit
import MachO
import CoreML

// MARK: RegressionStats

class RegressionStats: JSONCodable {
    let testInfo: TestInfo
    let memoryStats: MemoryStats
    let latencyStats: LatencyStats
    let staticAttributes: StaticAttributes
    let systemMeasurements: SystemMeasurements

    init(testInfo: TestInfo,
         memoryStats: MemoryStats,
         latencyStats: LatencyStats,
         staticAttributes: StaticAttributes,
         systemMeasurements: SystemMeasurements
    ) {
        self.testInfo = testInfo
        self.memoryStats = memoryStats
        self.latencyStats = latencyStats
        self.staticAttributes = staticAttributes
        self.systemMeasurements = systemMeasurements
    }

    func jsonData() throws -> Data {
        return try JSONEncoder().encode(self)
    }
}

// MARK: TestInfo

class TestInfo: JSONCodable {
    let device, audioFile: String
    let model: String
    let date: String
    let timeElapsedInSeconds: TimeInterval
    let timings: TranscriptionTimings?
    let transcript: String?
    let wer: Double

    init(device: String, audioFile: String, model: String, date: String, timeElapsedInSeconds: TimeInterval, timings: TranscriptionTimings?, transcript: String?, wer: Double) {
        self.device = device
        self.audioFile = audioFile
        self.model = model
        self.date = date
        self.timeElapsedInSeconds = timeElapsedInSeconds
        self.timings = timings
        self.transcript = transcript
        self.wer = wer
    }
}

// MARK: TestReport

struct TestReport: JSONCodable {
    let device: String
    let modelsTested: [String]
    let failureInfo: [String: String]

    init(device: String, modelsTested: [String], failureInfo: [String: String]) {
        self.device = device
        self.modelsTested = modelsTested
        self.failureInfo = failureInfo
    }
}

// MARK: Stats

class Stats: JSONCodable {
    var measurements: [Measurement]
    let units: String
    var totalNumberOfMeasurements: Int

    init(measurements: [Measurement], units: String, totalNumberOfMeasurements: Int) {
        self.measurements = measurements
        self.units = units
        self.totalNumberOfMeasurements = totalNumberOfMeasurements
    }

    func measure(from values: [Float], timeElapsed: TimeInterval) {
        var measurement: Measurement
        if let min = values.min(), let max = values.max() {
            measurement = Measurement(
                min: min,
                max: max,
                average: values.reduce(0,+) / Float(values.count),
                numberOfMeasurements: values.count,
                timeElapsed: timeElapsed
            )
            self.measurements.append(measurement)
            self.totalNumberOfMeasurements += values.count
        }
    }
}

// MARK: StaticAttributes
class StaticAttributes: Codable{
    let osVersion: String
    let isLowPowerMode: String
    let encoderCompute: String
    let decoderCompute: String
    
    init(encoderCompute: MLComputeUnits, decoderCompute: MLComputeUnits){
        let version = ProcessInfo.processInfo.operatingSystemVersion
        self.osVersion = "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        self.isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled ? "Enabled" : "Disabled"
        self.encoderCompute = encoderCompute.stringValue
        self.decoderCompute = decoderCompute.stringValue
    }
}

class SystemMeasurements: Codable{
    let systemMemory: [SystemMemoryUsage]
    let diskSpace: [DiskSpace]
    let batteryLevel: [Float]
    let timeElapsed: [TimeInterval]
    
    init(systemMemory: [SystemMemoryUsage], diskSpace: [DiskSpace], batteryLevel: [Float], timeElapsed: [TimeInterval]) {
        self.systemMemory = systemMemory
        self.diskSpace = diskSpace
        self.batteryLevel = batteryLevel
        self.timeElapsed = timeElapsed
    }
}

// MARK: LatencyStats

class LatencyStats: Stats {
    override init(measurements: [Measurement] = [], units: String, totalNumberOfMeasurements: Int = 0) {
        super.init(measurements: measurements, units: units, totalNumberOfMeasurements: totalNumberOfMeasurements)
    }

    required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    func calculate(from total: Double, runs: Int) -> Double {
        return runs > 0 ? total / Double(runs) : -1
    }
}

class MemoryStats: Stats {
    var preTranscribeMemory: Float
    var postTranscribeMemory: Float

    init(measurements: [Measurement] = [], units: String, totalNumberOfMeasurements: Int = 0, preTranscribeMemory: Float, postTranscribeMemory: Float) {
        self.preTranscribeMemory = preTranscribeMemory
        self.postTranscribeMemory = postTranscribeMemory
        super.init(measurements: measurements, units: units, totalNumberOfMeasurements: totalNumberOfMeasurements)
    }

    required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    /// Implement the encode(to:) method
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try super.encode(to: encoder)
        try container.encode(preTranscribeMemory, forKey: .preTranscribeMemory)
        try container.encode(postTranscribeMemory, forKey: .postTranscribeMemory)
    }

    /// Coding keys for MemoryStats properties
    enum CodingKeys: String, CodingKey {
        case preTranscribeMemory
        case postTranscribeMemory
    }
}

struct Measurement: JSONCodable {
    let min, max, average: Float
    let numberOfMeasurements: Int
    let timeElapsed: TimeInterval
}

protocol JSONCodable: Codable {}

extension JSONCodable {
    func jsonData() throws -> Data {
        return try JSONEncoder().encode(self)
    }
}

extension Data {
    var prettyPrintedJSONString: NSString? { // NSString gives us a nice sanitized debugDescription
        guard let object = try? JSONSerialization.jsonObject(with: self, options: []),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let prettyPrintedString = NSString(data: data, encoding: String.Encoding.utf8.rawValue) else { return nil }

        return prettyPrintedString
    }
}

// MARK: - SystemMemoryChecker

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
class AppMemoryChecker: NSObject {
    static func getMemoryUsed() -> UInt64 {
        // The `TASK_VM_INFO_COUNT` and `TASK_VM_INFO_REV1_COUNT` macros are too
        // complex for the Swift C importer, so we have to define them ourselves.
        let TASK_VM_INFO_COUNT = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        guard let offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else { return 0 }
        let TASK_VM_INFO_REV1_COUNT = mach_msg_type_number_t(offset / MemoryLayout<integer_t>.size)
        var info = task_vm_info_data_t()
        var count = TASK_VM_INFO_COUNT
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard
            kr == KERN_SUCCESS,
            count >= TASK_VM_INFO_REV1_COUNT
        else { return 0 }

        let usedBytes = Float(info.phys_footprint)
        let usedBytesInt = UInt64(usedBytes)
        let usedMB = usedBytesInt / 1024 / 1024
        return usedMB
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
class SystemMemoryCheckerAdvanced: NSObject {

    static func getMemoryUsage() -> SystemMemoryUsage {
        // Get total and available memory using host_statistics64
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: stats) / MemoryLayout<integer_t>.size)
        let hostPort = mach_host_self()
        let result = withUnsafeMutablePointer(to: &stats) { statsPtr -> kern_return_t in
            statsPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(hostPort, HOST_VM_INFO64, intPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else {
            return SystemMemoryUsage(totalAvailableGB: 0, totalUsedGB: 0, appAllocatedGB: 0, appUsedGB: 0, swapUsedGB: 0)
        }

        let pageSize = UInt64(vm_kernel_page_size)
        let totalMemory = Float(ProcessInfo.processInfo.physicalMemory) / 1024 / 1024 / 1024
        let freeMemory = Float(stats.free_count) * Float(pageSize) / 1024 / 1024 / 1024
        let inactiveMemory = Float(stats.inactive_count) * Float(pageSize) / 1024 / 1024 / 1024
        let availableMemory = freeMemory + inactiveMemory
        let activeMemory = Float(stats.active_count) * Float(pageSize) / 1024 / 1024 / 1024
        let wiredMemory = Float(stats.wire_count) * Float(pageSize) / 1024 / 1024 / 1024
        let usedMemory = totalMemory - availableMemory

        // Get task-specific memory footprint using task_info
        let TASK_VM_INFO_COUNT = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        guard let offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else {
            return SystemMemoryUsage(totalAvailableGB: 0, totalUsedGB: 0, appAllocatedGB: 0, appUsedGB: 0, swapUsedGB: 0)
        }
        let TASK_VM_INFO_REV1_COUNT = mach_msg_type_number_t(offset / MemoryLayout<integer_t>.size)
        var info = task_vm_info_data_t()
        var countInfo = TASK_VM_INFO_COUNT
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(countInfo)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &countInfo)
            }
        }

        guard
            kr == KERN_SUCCESS,
            countInfo >= TASK_VM_INFO_REV1_COUNT
        else {
            return SystemMemoryUsage(totalAvailableGB: 0, totalUsedGB: 0, appAllocatedGB: 0, appUsedGB: 0, swapUsedGB: 0)
        }

        let appAllocatedBytes = UInt64(info.phys_footprint)
        let appAllocatedGB = Float(appAllocatedBytes) / 1024 / 1024 / 1024

        let appUsedBytes = UInt64(info.resident_size)
        let appUsedGB = Float(appUsedBytes) / 1024 / 1024 / 1024

        // Get swap memory usage
        let swapUsedBytes = UInt64(stats.swapouts) * pageSize
        let swapUsedGB = Float(swapUsedBytes) / 1024 / 1024 / 1024

        return SystemMemoryUsage(totalAvailableGB: availableMemory, totalUsedGB: usedMemory, appAllocatedGB: appAllocatedGB, appUsedGB: appUsedGB, swapUsedGB: swapUsedGB)
    }
}

import Foundation
#if canImport(UIKit)
import UIKit
#endif

#if canImport(IOKit)
import IOKit.ps
#endif

class BatteryLevelChecker: NSObject {
    static func getBatteryLevel() -> Float? {
        #if os(iOS) || os(visionOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let batteryLevel = UIDevice.current.batteryLevel
        UIDevice.current.isBatteryMonitoringEnabled = false
        return batteryLevel >= 0 ? batteryLevel * 100 : nil
        #elseif os(watchOS)
        let batteryLevel = WKInterfaceDevice.current().batteryLevel
        return batteryLevel >= 0 ? batteryLevel * 100 : nil
        #elseif os(macOS)
        return getMacOSBatteryLevel()
        #else
        return nil
        #endif
    }
    
    #if os(macOS)
    private static func getMacOSBatteryLevel() -> Float? {
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as [CFTypeRef]
        for ps in sources {
            if let description = IOPSGetPowerSourceDescription(snapshot, ps).takeUnretainedValue() as? [String: Any] {
                if let currentCapacity = description[kIOPSCurrentCapacityKey] as? Int,
                   let maxCapacity = description[kIOPSMaxCapacityKey] as? Int {
                    return (Float(currentCapacity) / Float(maxCapacity)) * 100
                }
            }
        }
        return nil
    }
    #endif
}

struct DiskSpace: Codable {
    let totalSpaceGB: Float?
    let freeSpaceGB: Float?
}

struct SystemMemoryUsage: Codable {
    let totalAvailableGB: Float
    let totalUsedGB: Float
    let appAllocatedGB: Float
    let appUsedGB: Float
    let swapUsedGB: Float
}

class DiskSpaceChecker: NSObject {
    static func getDiskSpace() -> DiskSpace {
        #if os(iOS) || os(watchOS) || os(visionOS)
        return getMobileOSDiskSpace()
        #elseif os(macOS)
        return getMacOSDiskSpace()
        #else
        return DiskSpace(totalSpaceGB: nil, freeSpaceGB: nil)
        #endif
    }
    
    #if os(iOS) || os(watchOS) || os(visionOS)
    private static func getMobileOSDiskSpace() -> DiskSpace {
        let fileManager = FileManager.default
        do {
            let attributes = try fileManager.attributesOfFileSystem(forPath: NSHomeDirectory())
            if let totalSpace = attributes[.systemSize] as? NSNumber,
               let freeSpace = attributes[.systemFreeSize] as? NSNumber {
                return DiskSpace(
                    totalSpaceGB: Float(truncating: totalSpace) / 1024 / 1024 / 1024,
                    freeSpaceGB: Float(truncating: freeSpace) / 1024 / 1024 / 1024
                )
            }
        } catch {
            print("Error retrieving file system attributes: \(error)")
        }
        return DiskSpace(totalSpaceGB: nil, freeSpaceGB: nil)
    }
    #endif
    
    #if os(macOS)
    private static func getMacOSDiskSpace() -> DiskSpace {
        let fileManager = FileManager.default
        do {
            let homeDirectory = fileManager.homeDirectoryForCurrentUser
            let attributes = try fileManager.attributesOfFileSystem(forPath: homeDirectory.path)
            if let totalSpace = attributes[.systemSize] as? NSNumber,
               let freeSpace = attributes[.systemFreeSize] as? NSNumber {
                return DiskSpace(
                    totalSpaceGB: Float(truncating: totalSpace) / 1024 / 1024 / 1024,
                    freeSpaceGB: Float(truncating: freeSpace) / 1024 / 1024 / 1024
                )
            }
        } catch {
            print("Error retrieving file system attributes: \(error)")
        }
        return DiskSpace(totalSpaceGB: nil, freeSpaceGB: nil)
    }
    #endif
}



private extension MLComputeUnits{
    var stringValue: String {
        switch self {
        case .cpuOnly:
            return "CPU Only"
        case .cpuAndGPU:
            return "CPU and GPU"
        case .all:
            return "All"
        case .cpuAndNeuralEngine:
            return "CPU and Neural Engine"
        @unknown default:
            return "Unknown"
        }
    }
}
