import SwiftUI

struct RotaryKnob: View {
  let label: String
  @Binding var value: Double
  let range: ClosedRange<Double>
  let defaultValue: Double
  var unit = ""
  var precision = 2
  var diameter: CGFloat = 72
  var accent: Color = .teal
  var logarithmic = false
  var onEditingChanged: (Bool) -> Void = { _ in }

  @State private var dragOrigin: Double?

  private var normalized: Double {
    normalize(value)
  }

  var body: some View {
    VStack(spacing: 4) {
      Text(label)
        .font(.caption2)
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .minimumScaleFactor(0.75)
      Canvas { context, size in
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let radius = min(size.width, size.height) / 2 - 3
        let track = Path { path in
          path.addArc(center: center, radius: radius - 6, startAngle: .degrees(135), endAngle: .degrees(405), clockwise: false)
        }
        context.stroke(track, with: .color(.secondary.opacity(0.22)), style: StrokeStyle(lineWidth: 3, lineCap: .round))
        let active = Path { path in
          path.addArc(
            center: center,
            radius: radius - 6,
            startAngle: .degrees(135),
            endAngle: .degrees(135 + 270 * normalized),
            clockwise: false
          )
        }
        context.stroke(active, with: .color(accent), style: StrokeStyle(lineWidth: 4, lineCap: .round))
        let angle = (135 + 270 * normalized) * .pi / 180
        var pointer = Path()
        pointer.move(to: CGPoint(x: center.x + cos(angle) * (radius - 17), y: center.y + sin(angle) * (radius - 17)))
        pointer.addLine(to: CGPoint(x: center.x + cos(angle) * (radius - 7), y: center.y + sin(angle) * (radius - 7)))
        context.stroke(pointer, with: .color(.primary), style: StrokeStyle(lineWidth: 3, lineCap: .round))
      }
      .frame(width: diameter, height: diameter)
      .background(Circle().fill(.secondary.opacity(0.08)))
      .overlay(Circle().stroke(.secondary.opacity(0.18)))
      .contentShape(Circle())
      .gesture(
        DragGesture(minimumDistance: 0)
          .onChanged { gesture in
            if dragOrigin == nil {
              dragOrigin = value
              onEditingChanged(true)
            }
            let fine = NSEvent.modifierFlags.contains(.shift) ? 0.1 : 1.0
            let start = normalize(dragOrigin ?? value)
            value = denormalize(min(max(start - gesture.translation.height / 200 * fine, 0), 1))
          }
          .onEnded { _ in
            dragOrigin = nil
            onEditingChanged(false)
          }
      )
      .onTapGesture(count: 2) {
        onEditingChanged(true)
        value = defaultValue
        onEditingChanged(false)
      }
      Text(String(format: "%.*f%@%@", precision, value, unit.isEmpty ? "" : " ", unit))
        .font(.system(.caption, design: .monospaced))
        .lineLimit(1)
    }
    .frame(minWidth: diameter + 18)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(label)
    .accessibilityValue("\(value) \(unit)")
    .accessibilityAdjustableAction { direction in
      onEditingChanged(true)
      let step = (range.upperBound - range.lowerBound) / 100
      switch direction {
      case .increment: value = min(value + step, range.upperBound)
      case .decrement: value = max(value - step, range.lowerBound)
      @unknown default: break
      }
      onEditingChanged(false)
    }
  }

  private func normalize(_ input: Double) -> Double {
    if logarithmic {
      let low = log10(max(range.lowerBound, 1e-9))
      let high = log10(max(range.upperBound, 1e-9))
      return (log10(max(input, 1e-9)) - low) / (high - low)
    }
    return (input - range.lowerBound) / (range.upperBound - range.lowerBound)
  }

  private func denormalize(_ input: Double) -> Double {
    if logarithmic {
      let low = log10(max(range.lowerBound, 1e-9))
      let high = log10(max(range.upperBound, 1e-9))
      return pow(10, low + (high - low) * input)
    }
    return range.lowerBound + (range.upperBound - range.lowerBound) * input
  }
}
