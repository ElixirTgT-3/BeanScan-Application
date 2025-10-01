import 'package:flutter/material.dart';
import '../utils/app_colors.dart';

/// Draws a coffee bean outline with a center "crack" line.
/// The [severityLevel] controls how jagged/long the crack is:
/// 1 = Mild, 2 = Moderate, 3 = Severe
class BeanSeverityIcon extends StatelessWidget {
  final int severityLevel; // 1..3
  final double size;
  final Color color;

  const BeanSeverityIcon({
    super.key,
    required this.severityLevel,
    this.size = 72,
    this.color = AppColors.primaryBrown,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size.square(size),
      painter: _BeanPainter(severityLevel: severityLevel, color: color),
    );
  }
}

class _BeanPainter extends CustomPainter {
  final int severityLevel;
  final Color color;

  _BeanPainter({required this.severityLevel, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final width = size.width;
    final height = size.height;
    final rect = Rect.fromLTWH(width * 0.12, height * 0.12, width * 0.76, height * 0.76);

    // Bean outline: two arcs to mimic an ellipse with slight pinch
    final outline = Path();
    outline.addRRect(RRect.fromRectXY(rect, width * 0.45, height * 0.45));
    canvas.drawPath(outline, stroke);

    // Central split curve (crack) – severity controls jaggedness/amplitude
    final crack = Path();
    final leftX = rect.center.dx - rect.width * 0.08;
    final rightX = rect.center.dx + rect.width * 0.08;

    // Build a polyline from top to bottom zig-zagging slightly
    final steps = severityLevel == 1 ? 3 : (severityLevel == 2 ? 5 : 7);
    final amp = severityLevel == 1 ? rect.width * 0.05 : (severityLevel == 2 ? rect.width * 0.08 : rect.width * 0.12);

    double y = rect.top + rect.height * 0.1;
    final yStep = (rect.height * 0.8) / steps;
    bool toRight = true;

    crack.moveTo(leftX, y);
    for (int i = 0; i < steps; i++) {
      y += yStep;
      final x = rect.center.dx + (toRight ? amp : -amp);
      crack.lineTo(x, y);
      toRight = !toRight;
    }
    crack.lineTo(rightX, rect.bottom - rect.height * 0.1);

    canvas.drawPath(crack, stroke);
  }

  @override
  bool shouldRepaint(covariant _BeanPainter oldDelegate) {
    return oldDelegate.severityLevel != severityLevel || oldDelegate.color != color;
  }
}


