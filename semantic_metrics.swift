import Foundation
import NaturalLanguage

struct Metrics {
    let precision: Double
    let coverage: Double
    let f1: Double
    let diversity: Double
}

func mean(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0.0 }
    return values.reduce(0.0, +) / Double(values.count)
}

func clampedCosine(_ left: [Double], _ right: [Double]) -> Double {
    guard !left.isEmpty, left.count == right.count else { return 0.0 }
    var dot = 0.0
    for index in left.indices { dot += left[index] * right[index] }
    return max(0.0, min(1.0, dot))
}

let inputData = FileHandle.standardInput.readDataToEndOfFile()
guard
    let input = try JSONSerialization.jsonObject(with: inputData) as? [String: Any],
    let rows = input["rows"] as? [[String: Any]],
    let modelNames = input["models"] as? [String],
    let groupNames = input["groups"] as? [String],
    let wordEmbedding = NLEmbedding.wordEmbedding(for: .english)
else {
    fputs("Unable to load input or the English embedding model.\n", stderr)
    exit(1)
}

var embeddingCache: [String: [Double]] = [:]

func sentenceVector(_ text: String) -> [Double] {
    if let cached = embeddingCache[text] { return cached }
    let tokenizer = NLTokenizer(unit: .word)
    tokenizer.string = text
    var accumulator = Array(repeating: 0.0, count: wordEmbedding.dimension)
    var found = 0
    tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
        let token = String(text[range]).lowercased()
        if let vector = wordEmbedding.vector(for: token) {
            for index in accumulator.indices { accumulator[index] += vector[index] }
            found += 1
        }
        return true
    }
    if found > 0 {
        let divisor = Double(found)
        for index in accumulator.indices { accumulator[index] /= divisor }
        let norm = sqrt(accumulator.reduce(0.0) { $0 + $1 * $1 })
        if norm > 0 {
            for index in accumulator.indices { accumulator[index] /= norm }
        }
    }
    embeddingCache[text] = accumulator
    return accumulator
}

func calculate(candidateQuestions: [String], referenceQuestions: [String]) -> Metrics {
    let candidates = candidateQuestions.map(sentenceVector)
    let references = referenceQuestions.map(sentenceVector)

    let precision = mean(candidates.map { candidate in
        references.map { clampedCosine(candidate, $0) }.max() ?? 0.0
    })
    let coverage = mean(references.map { reference in
        candidates.map { clampedCosine(reference, $0) }.max() ?? 0.0
    })
    let f1 = (precision + coverage) > 0
        ? (2.0 * precision * coverage / (precision + coverage))
        : 0.0

    var pairwise: [Double] = []
    if candidates.count >= 2 {
        for left in 0..<(candidates.count - 1) {
            for right in (left + 1)..<candidates.count {
                pairwise.append(clampedCosine(candidates[left], candidates[right]))
            }
        }
    }
    let diversity = max(0.0, min(1.0, 1.0 - mean(pairwise)))
    return Metrics(precision: precision, coverage: coverage, f1: f1, diversity: diversity)
}

let metricDefinitions: [(key: String, label: String, value: (Metrics) -> Double)] = [
    ("precision", "Semantic Precision", { $0.precision }),
    ("coverage", "Teacher Coverage", { $0.coverage }),
    ("f1", "Semantic F1", { $0.f1 }),
    ("diversity", "Diversity", { $0.diversity }),
]

var headers: [String] = []
for metric in metricDefinitions {
    for group in groupNames { headers.append("\(metric.label) - \(group)") }
}

var outputMatrix: [[String]] = []
var rawScores: [String: [Double]] = Dictionary(
    uniqueKeysWithValues: metricDefinitions.map { ($0.key, []) }
)
var diagnostics: [[String: Any]] = []

for row in rows {
    let sheetRow = row["sheetRow"] as? Int ?? 0
    let subject = row["subject"] as? String ?? ""
    let referenceQuestions = row["referenceQuestions"] as? [String] ?? []
    let groups = row["groups"] as? [[String: Any]] ?? []
    var byGroup: [[[String: Double]]] = []

    for group in groups {
        let models = group["models"] as? [[String: Any]] ?? []
        var modelMetrics: [[String: Double]] = []
        for model in models {
            let questions = model["questions"] as? [String] ?? []
            let score = calculate(candidateQuestions: questions, referenceQuestions: referenceQuestions)
            modelMetrics.append([
                "precision": score.precision,
                "coverage": score.coverage,
                "f1": score.f1,
                "diversity": score.diversity,
            ])
        }
        byGroup.append(modelMetrics)
    }

    var outputRow: [String] = []
    for metric in metricDefinitions {
        for groupIndex in groupNames.indices {
            var lines: [String] = []
            for modelIndex in modelNames.indices {
                let value = byGroup[groupIndex][modelIndex][metric.key] ?? 0.0
                rawScores[metric.key, default: []].append(value)
                lines.append("\(modelNames[modelIndex]) - \(String(format: "%.4f", value))")
            }
            outputRow.append(lines.joined(separator: "\n"))
        }
    }
    outputMatrix.append(outputRow)
    diagnostics.append([
        "sheetRow": sheetRow,
        "subject": subject,
        "referenceQuestionCount": referenceQuestions.count,
    ])
}

var summary: [String: Double] = [:]
for metric in metricDefinitions {
    summary[metric.key] = mean(rawScores[metric.key] ?? [])
}

let output: [String: Any] = [
    "embedding": input["embedding"] as? String ?? "",
    "headers": headers,
    "matrix": outputMatrix,
    "summary": summary,
    "diagnostics": diagnostics,
    "scoreCellCount": outputMatrix.count * headers.count,
    "individualScoreCount": outputMatrix.count * headers.count * modelNames.count,
]
let outputData = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
FileHandle.standardOutput.write(outputData)
