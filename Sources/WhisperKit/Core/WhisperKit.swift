// For licensing see accompanying LICENSE.md file.
// Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreML
import Foundation
import Hub
import TensorUtils
import Tokenizers

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class WhisperKit {
    
    // MARK: - Model Configuration
    public private(set) var modelVariant: ModelVariant = .tiny
    public private(set) var modelState: ModelState = .unloaded
    public var modelCompute: ModelComputeOptions
    public var tokenizer: WhisperTokenizer?

    // MARK: - Processing Components
    public var audioProcessor: any AudioProcessing
    public var featureExtractor: any FeatureExtracting
    public var audioEncoder: any AudioEncoding
    public var textDecoder: any TextDecoding
    public var logitsFilters: [any LogitsFiltering]
    public var segmentSeeker: any SegmentSeeking

    // MARK: - Default Model Parameters
    public static let sampleRate: Int = 16000
    public static let hopLength: Int = 160
    public static let chunkLength: Int = 30 // seconds
    public static let windowSamples: Int = sampleRate * chunkLength
    public static let secondsPerTimeToken: Float = 0.02

    // MARK: - Progress Tracking
    public private(set) var currentTimings: TranscriptionTimings
    public let progress = Progress()

    // MARK: - Model Storage
    public var modelFolder: URL?
    public var tokenizerFolder: URL?
    public let useBackgroundDownloadSession: Bool

    // MARK: - Initialization
    public init(
        model: String? = nil,
        downloadBase: URL? = nil,
        modelRepo: String? = nil,
        modelFolder: String? = nil,
        tokenizerFolder: URL? = nil,
        computeOptions: ModelComputeOptions? = nil,
        audioProcessor: (any AudioProcessing)? = nil,
        featureExtractor: (any FeatureExtracting)? = nil,
        audioEncoder: (any AudioEncoding)? = nil,
        textDecoder: (any TextDecoding)? = nil,
        logitsFilters: [any LogitsFiltering]? = nil,
        segmentSeeker: (any SegmentSeeking)? = nil,
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        prewarm: Bool? = nil,
        load: Bool? = nil,
        download: Bool = true,
        useBackgroundDownloadSession: Bool = false
    ) async throws {
        self.modelCompute = computeOptions ?? ModelComputeOptions()
        self.audioProcessor = audioProcessor ?? AudioProcessor()
        self.featureExtractor = featureExtractor ?? FeatureExtractor()
        self.audioEncoder = audioEncoder ?? AudioEncoder()
        self.textDecoder = textDecoder ?? TextDecoder()
        self.logitsFilters = logitsFilters ?? []
        self.segmentSeeker = segmentSeeker ?? SegmentSeeker()
        self.tokenizerFolder = tokenizerFolder
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        self.currentTimings = TranscriptionTimings()
        Logging.shared.logLevel = verbose ? logLevel : .none

        try await setupModels(model: model, downloadBase: downloadBase, modelRepo: modelRepo, modelFolder: modelFolder, download: download)

        if prewarm == true {
            Logging.info("Prewarming models...")
            try await prewarmModels()
        }

        if load ?? (modelFolder != nil) {
            Logging.info("Loading models...")
            try await loadModels()
        }
    }

    // MARK: - Model Handling

    public static func recommendedModels() -> (default: String, disabled: [String]) {
        let deviceName = Self.deviceName()
        Logging.debug("Running on \(deviceName)")

        let defaultModel = modelSupport(for: deviceName).default
        let disabledModels = modelSupport(for: deviceName).disabled
        return (defaultModel, disabledModels)
    }

    public static func deviceName() -> String {
        var utsname = utsname()
        uname(&utsname)
        return withUnsafePointer(to: &utsname.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
    }

    public static func fetchAvailableModels(from repo: String = "argmaxinc/whisperkit-coreml", matching patterns: [String] = ["openai_*", "distil-whisper_*"]) async throws -> [String] {
        let hubApi = HubApi()
        let modelFiles = try await hubApi.getFilenames(from: repo, matching: patterns)
        return formatModelFiles(modelFiles)
    }

    public static func formatModelFiles(_ modelFiles: [String]) -> [String] {
        let modelFilters = ModelVariant.allCases.map { "\($0.description)\($0.description.contains("large") ? "" : "/")" }
        let modelVariants = modelFiles.map { $0.components(separatedBy: "/").first! + "/" }
        let filteredVariants = Set(modelVariants.filter { modelFilters.contains(where: $0.contains) })
        return filteredVariants.sorted()
    }

    public static func download(
        variant: String,
        from: String = "argmaxinc/whisperkit-coreml",
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let hubApi = HubApi()
        let modelFolder = try await hubApi.snapshot(from: repo, matching: ["*\(variant)/*"]) { progress in
            progressCallback?(progress)
        }
        return modelFolder
    }

    private func setupModels(
        model: String?,
        downloadBase: URL? = nil,
        modelRepo: String?,
        modelFolder: String?,
        download: Bool
    ) async throws {
        let modelVariant = model ?? WhisperKit.recommendedModels().default

        if let folder = modelFolder {
            self.modelFolder = URL(fileURLWithPath: folder)
        } else if download {
            let repo = modelRepo ?? "argmaxinc/whisperkit-coreml"
            self.modelFolder = try await Self.download(variant: modelVariant, from: repo)
        }
    }

    public func prewarmModels() async throws {
        try await loadModels(prewarmMode: true)
    }

    nonisolated func reloadModels(
        modelCompute: ModelComputeOptions = .init(),
        prewarmMode: PrewarmMode = .all
    ) async throws {
        let modelVariant = self.modelVariant
        let model = self.model
        let tokenizer = self.tokenizer

        let (featureExtractor, audioEncoder, textDecoder): (any FeatureExtracting, any AudioEncoding, any TextDecoding) =
        try await Self.load(
            modelVariant: modelVariant,
            model: model,
            tokenizer: tokenizer,
            modelCompute: modelCompute,
            downloadModel: download,
            prewarmMode: prewarmMode
        )

        self.featureExtractor = featureExtractor
        self.audioEncoder = audioEncoder
        self.textDecoder = textDecoder
    }
    
    static func load(
      modelVariant: ModelVariant,
      model: String,
      tokenizer: Tokenizer,
      modelCompute: ModelComputeOptions = .init(),
      downloadModel: Bool,
      prewarmMode: PrewarmMode = .all
    ) async throws -> (any FeatureExtracting, any AudioEncoding, any TextDecoding) {
      let (logmelUrl, encoderUrl, decoderUrl) = try await getModelFileURLs(downloadModel: downloadModel, model: model, variant: modelVariant)

      let featureExtractor = FeatureExtractor()
      let audioEncoder = try AudioEncoder(modelVariant: modelVariant)
      let textDecoder = try TextDecoder(modelVariant: modelVariant, tokenizer: tokenizer)

        // try await featureExtractor.loadModel(at: logmelUrl, computeUnits: modelCompute.melCompute, prewarmMode: prewarmMode)
        // try await audioEncoder.loadModel(at: encoderUrl, computeUnits: modelCompute.audioEncoderCompute, prewarmMode: prewarmMode)
        // try await textDecoder.loadModel(at: decoderUrl, computeUnits: modelCompute.textDecoderCompute, prewarmMode: prewarmMode)

      return (featureExtractor, audioEncoder, textDecoder)
    }

    public func loadModels(prewarmMode: Bool = false) async throws {
        modelState = prewarmMode ? .prewarming : .loading
        guard let path = modelFolder else { throw WhisperError.modelsUnavailable("Model folder is not set.") }

        let logmelUrl = path.appending(path: "MelSpectrogram.mlmodelc")
        let encoderUrl = path.appending(path: "AudioEncoder.mlmodelc")
        let decoderUrl = path.appending(path: "TextDecoder.mlmodelc")

        for url in [logmelUrl, encoderUrl, decoderUrl] where !FileManager.default.fileExists(atPath: url.path) {
            throw WhisperError.modelsUnavailable("Model file not found at \(url.path)")
        }

        try await reloadModels(modelCompute: modelCompute, prewarmMode: prewarmMode)
        modelState = prewarmMode ? .prewarmed : .loaded
        Logging.info("Loaded models for whisper size: \(modelVariant)")
    }

    public func unloadModels() {
        modelState = .unloading
        [featureExtractor, audioEncoder, textDecoder].forEach { ($0 as? WhisperMLModel)?.unloadModel() }
        modelState = .unloaded
    }
} 