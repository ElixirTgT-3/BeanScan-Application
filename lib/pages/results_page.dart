import 'package:flutter/material.dart';
import 'dart:io';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';

class ResultsPage extends StatelessWidget {
  final BeanPrediction prediction;
  final String imagePath;
  final Map<String, dynamic>? defectDetection;

  const ResultsPage({
    super.key,
    required this.prediction,
    required this.imagePath,
    this.defectDetection,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.scanDarkGrey,
      appBar: AppBar(
        backgroundColor: AppColors.scanDarkGrey,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Scanned Coffee Bean Result',
          style: TextStyle(color: Colors.white),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(AppConstants.largePadding),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildImagePreview(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildInfoCard(),
              const SizedBox(height: AppConstants.largeSpacing),
              if (defectDetection != null) ...[
                _buildDefectDetectionCard(),
                const SizedBox(height: AppConstants.largeSpacing),
              ],
              _buildHealthTiles(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildYesNoButtons(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Container(
      height: 210,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.15),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
        border: Border.all(color: Colors.white24),
      ),
      clipBehavior: Clip.antiAlias,
      child: imagePath.isNotEmpty
          ? Stack(
              children: [
                // Base image
                Image.file(
                  File(imagePath), 
                  fit: BoxFit.cover,
                  width: double.infinity,
                  height: double.infinity,
                ),
                // Defect annotations overlay
                if (defectDetection != null && defectDetection!['detections'] != null)
                  _buildDefectAnnotations(),
                // Fallback: Show defect count if no visual annotations work
                if (defectDetection != null && defectDetection!['summary'] != null)
                  _buildDefectCountOverlay(),
              ],
            )
          : const Center(
              child: Icon(Icons.image, color: Colors.white54, size: 48),
            ),
    );
  }

  Widget _buildDefectAnnotations() {
    final detections = defectDetection!['detections'] as List<dynamic>?;
    if (detections == null || detections.isEmpty) {
      return const SizedBox.shrink();
    }

    // Debug: Print detection data
    print('Defect detections: ${detections.length}');
    for (int i = 0; i < detections.length; i++) {
      print('Detection $i: ${detections[i]}');
    }

    return CustomPaint(
      painter: DefectAnnotationPainter(detections),
      size: Size.infinite,
    );
  }

  Widget _buildDefectCountOverlay() {
    final summary = defectDetection!['summary'] as Map<String, dynamic>?;
    if (summary == null) return const SizedBox.shrink();

    final totalDefects = summary['total_defects'] as int? ?? 0;
    if (totalDefects == 0) return const SizedBox.shrink();

    return Positioned(
      top: 10,
      right: 10,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.red.withOpacity(0.8),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          '$totalDefects Defects',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }

  Widget _buildInfoCard() {
    final DateTime now = DateTime.now();
    final String dateStr = '${now.month}/${now.day}/${now.year} - ${now.hour}:${now.minute.toString().padLeft(2, '0')}';
    final double healthyPct = (prediction.confidence * 100).clamp(0, 100);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.85),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(dateStr, style: const TextStyle(color: AppColors.textDarkGrey)),
              const Icon(Icons.refresh, size: 18, color: AppColors.textDarkGrey),
            ],
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          Text('Type: ${prediction.prediction}', style: const TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 4),
          const Text('Mold: - | Bleached: -', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 4),
          const Text('Total Beans: -', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: AppConstants.mediumSpacing),
          const Text('Estimated Shelf Life', style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey)),
          const SizedBox(height: 2),
          const Text('Confidence Score:', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 6),
          Row(
            children: [
              _circularScore(label: 'Confidence', percent: healthyPct, color: AppColors.primaryBrown),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildDefectDetectionCard() {
    if (defectDetection == null) return const SizedBox.shrink();
    
    final summary = defectDetection!['summary'];
    final detections = defectDetection!['detections'] as List<dynamic>? ?? [];
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.85),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.bug_report, color: AppColors.primaryBrown, size: 20),
              const SizedBox(width: 8),
              Text(
                'Defect Detection Results',
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDarkGrey,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          
          // Quality Grade
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Quality Grade:', style: TextStyle(color: AppColors.textDarkGrey)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: _getQualityColor(summary['quality_grade']),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  summary['quality_grade'] ?? 'Unknown',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // Total Defects
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Total Defects:', style: TextStyle(color: AppColors.textDarkGrey)),
              Text(
                '${summary['total_defects'] ?? 0}',
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDarkGrey,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // Defect Percentage
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Defect Area:', style: TextStyle(color: AppColors.textDarkGrey)),
              Text(
                '${summary['defect_percentage']?.toStringAsFixed(1) ?? '0.0'}%',
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDarkGrey,
                ),
              ),
            ],
          ),
          
          // Defect Types
          if (summary['defect_types'] != null && (summary['defect_types'] as Map).isNotEmpty) ...[
            const SizedBox(height: AppConstants.mediumSpacing),
            const Text('Defect Types:', style: TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.textDarkGrey,
            )),
            const SizedBox(height: 4),
            Wrap(
              spacing: 8,
              runSpacing: 4,
              children: (summary['defect_types'] as Map<String, dynamic>).entries
                  .map((entry) => Chip(
                        label: Text('${entry.key}: ${entry.value}'),
                        backgroundColor: Colors.orange.withOpacity(0.2),
                        labelStyle: const TextStyle(fontSize: 12),
                      ))
                  .toList(),
            ),
          ],
        ],
      ),
    );
  }

  Color _getQualityColor(String grade) {
    switch (grade) {
      case 'A+':
      case 'A':
        return Colors.green;
      case 'B+':
      case 'B':
        return Colors.blue;
      case 'C+':
      case 'C':
        return Colors.orange;
      case 'D':
        return Colors.red;
      case 'F':
        return Colors.red[800]!;
      default:
        return Colors.grey;
    }
  }

  Widget _buildHealthTiles() {
    double healthy, defective;
    
    if (defectDetection != null && defectDetection!['summary'] != null) {
      final qualityScore = defectDetection!['summary']['quality_score'] ?? 0.0;
      healthy = (qualityScore * 100).clamp(0, 100);
      defective = (100 - healthy).clamp(0, 100);
    } else {
      healthy = (prediction.confidence * 100).clamp(0, 100);
      defective = (100 - healthy).clamp(0, 100);
    }
    
    return Row(
      children: [
        Expanded(child: _healthCard('Healthy:', healthy, Colors.green)),
        const SizedBox(width: AppConstants.mediumSpacing),
        Expanded(child: _healthCard('Defective:', defective, AppColors.primaryBrown)),
      ],
    );
  }

  Widget _healthCard(String title, double percent, Color color) {
    return Container(
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.85),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: AppConstants.smallSpacing),
          Center(child: _circularPercent(percent: percent, color: color)),
        ],
      ),
    );
  }

  Widget _circularPercent({required double percent, required Color color}) {
    return SizedBox(
      height: 90,
      width: 90,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CircularProgressIndicator(
            value: percent / 100.0,
            strokeWidth: 8,
            backgroundColor: Colors.grey[200],
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
          Text('${percent.toStringAsFixed(0)}%', style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _circularScore({required String label, required double percent, required Color color}) {
    return Row(
      children: [
        _circularPercent(percent: percent, color: color),
        const SizedBox(width: AppConstants.mediumSpacing),
        Text(label, style: const TextStyle(color: AppColors.textDarkGrey)),
      ],
    );
  }

  Widget _buildYesNoButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).pop(),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryBrown,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
              ),
            ),
            child: const Text('Yes'),
          ),
        ),
        const SizedBox(width: AppConstants.mediumSpacing),
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).maybePop(),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[300],
              foregroundColor: AppColors.textDarkGrey,
              padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
              ),
            ),
            child: const Text('No'),
          ),
        ),
      ],
    );
  }
}

class DefectAnnotationPainter extends CustomPainter {
  final List<dynamic> detections;
  
  DefectAnnotationPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    print('Painting defects on canvas size: $size');
    
    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0; // Make lines thicker for visibility

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];
      final coordinates = detection['coordinates'] as Map<String, dynamic>?;
      if (coordinates == null) {
        print('Detection $i: No coordinates found');
        continue;
      }

      final x1 = coordinates['x1'] as double? ?? 0.0;
      final y1 = coordinates['y1'] as double? ?? 0.0;
      final x2 = coordinates['x2'] as double? ?? 0.0;
      final y2 = coordinates['y2'] as double? ?? 0.0;
      final defectType = detection['defect_type'] as String? ?? 'Unknown';
      final confidence = detection['confidence'] as double? ?? 0.0;

      print('Detection $i: $defectType at ($x1, $y1, $x2, $y2) with confidence $confidence');

      // Ensure coordinates are within canvas bounds
      final clampedX1 = x1.clamp(0.0, size.width);
      final clampedY1 = y1.clamp(0.0, size.height);
      final clampedX2 = x2.clamp(0.0, size.width);
      final clampedY2 = y2.clamp(0.0, size.height);

      // Draw bounding box
      final rect = Rect.fromLTRB(clampedX1, clampedY1, clampedX2, clampedY2);
      canvas.drawRect(rect, paint);

      // Draw a circle at the center for better visibility
      final centerX = (clampedX1 + clampedX2) / 2;
      final centerY = (clampedY1 + clampedY2) / 2;
      canvas.drawCircle(Offset(centerX, centerY), 5.0, paint..style = PaintingStyle.fill);

      // Draw label background
      final labelText = '$defectType (${(confidence * 100).toInt()}%)';
      textPainter.text = TextSpan(
        text: labelText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );
      textPainter.layout();

      final labelRect = Rect.fromLTWH(
        clampedX1, 
        (clampedY1 - textPainter.height - 4).clamp(0.0, size.height - textPainter.height - 4),
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final labelPaint = Paint()
        ..color = Colors.red.withOpacity(0.8)
        ..style = PaintingStyle.fill;

      canvas.drawRect(labelRect, labelPaint);
      textPainter.paint(canvas, Offset(clampedX1 + 4, labelRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
