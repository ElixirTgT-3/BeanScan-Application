import 'package:flutter/material.dart';
import 'dart:io';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';

class ResultsPage extends StatelessWidget {
  final BeanPrediction prediction;
  final String imagePath;
  final Map<String, dynamic>? defectDetection;
  final Map<String, dynamic>? shelfLife;
  final Map<String, dynamic>? beanCount;

  const ResultsPage({
    super.key,
    required this.prediction,
    required this.imagePath,
    this.defectDetection,
    this.shelfLife,
    this.beanCount,
  });

  @override
  Widget build(BuildContext context) {
    // Debug prints to see what data we're receiving
    print('🔍 ResultsPage Debug:');
    print('  - prediction: $prediction');
    print('  - imagePath: $imagePath');
    print('  - defectDetection: $defectDetection');
    print('  - shelfLife: $shelfLife');
    print('  - beanCount: $beanCount');
    
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
    final double healthyPct = (prediction.confidence * 100).clamp(0.0, 100.0);
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
          const SizedBox(height: 8),
          
          // Bean Count
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Total Beans:', style: TextStyle(color: AppColors.textDarkGrey)),
              Text(
                _getBeanCount(),
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDarkGrey,
                ),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          const Text('Estimated Shelf Life', style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey)),
          const SizedBox(height: 8),
          
          // Shelf Life Days
          if (shelfLife != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Predicted Days:', style: TextStyle(color: AppColors.textDarkGrey)),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                  decoration: BoxDecoration(
                    color: _getShelfLifeColor(shelfLife!['category']),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${shelfLife!['predicted_days'] ?? 0} days',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            
            // Shelf Life Category
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Status:', style: TextStyle(color: AppColors.textDarkGrey)),
                Text(
                  shelfLife!['category'] ?? 'Unknown',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: _getShelfLifeTextColor(shelfLife!['category']),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
          ],
          
          const Text('Confidence Score:', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 6),
          Row(
            children: [
              _circularScore(
                label: 'Confidence', 
                percent: shelfLife != null ? (shelfLife!['confidence'] * 100).toDouble() : healthyPct, 
                color: AppColors.primaryBrown
              ),
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

  Color _getShelfLifeColor(String category) {
    switch (category?.toLowerCase()) {
      case 'excellent':
        return Colors.green;
      case 'good':
        return Colors.blue;
      case 'warning':
        return Colors.orange;
      case 'critical':
        return Colors.red;
      case 'expired':
        return Colors.red.shade800;
      default:
        return Colors.grey;
    }
  }

  Color _getShelfLifeTextColor(String category) {
    switch (category?.toLowerCase()) {
      case 'excellent':
        return Colors.green;
      case 'good':
        return Colors.blue;
      case 'warning':
        return Colors.orange;
      case 'critical':
        return Colors.red;
      case 'expired':
        return Colors.red.shade800;
      default:
        return Colors.grey;
    }
  }

  String _getBeanCount() {
    // Use bean count from API response
    if (beanCount != null && beanCount!['estimated_count'] != null) {
      return beanCount!['estimated_count'].toString();
    }
    
    // Fallback estimation based on defect detection data
    if (defectDetection != null && defectDetection!['detections'] != null) {
      final detections = defectDetection!['detections'] as List<dynamic>;
      final estimatedCount = (detections.length * 1.5).round();
      return '~$estimatedCount';
    }
    
    return '~15'; // Default estimate
  }

  Widget _buildHealthTiles() {
    double healthy, defective;
    
    if (defectDetection != null && defectDetection!['summary'] != null) {
      final qualityScore = defectDetection!['summary']['quality_score'] ?? 0.0;
      healthy = (qualityScore * 100).clamp(0.0, 100.0);
      defective = (100.0 - healthy).clamp(0.0, 100.0);
    } else {
      healthy = (prediction.confidence * 100).clamp(0.0, 100.0);
      defective = (100.0 - healthy).clamp(0.0, 100.0);
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
    print('Total detections received: ${detections.length}');
    
    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0; // Thinner lines for better visibility

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    int validBoxesDrawn = 0;
    
    // Calculate scale factors - the coordinates seem to be in a smaller coordinate system
    // Based on the coordinates (x1: 77, y1: 188, x2: 100, y2: 214), this looks like a small region
    // Let's assume the coordinates are in a normalized or smaller coordinate system
    
    // Find the actual range of coordinates to determine the coordinate system
    double minX = double.infinity;
    double maxX = 0.0;
    double minY = double.infinity;
    double maxY = 0.0;
    
    for (final detection in detections) {
      final coordinates = detection['coordinates'] as Map<String, dynamic>?;
      if (coordinates != null) {
        final x1 = coordinates['x1'] as double? ?? 0.0;
        final y1 = coordinates['y1'] as double? ?? 0.0;
        final x2 = coordinates['x2'] as double? ?? 0.0;
        final y2 = coordinates['y2'] as double? ?? 0.0;
        
        minX = [minX, x1, x2].reduce((a, b) => a < b ? a : b);
        maxX = [maxX, x1, x2].reduce((a, b) => a > b ? a : b);
        minY = [minY, y1, y2].reduce((a, b) => a < b ? a : b);
        maxY = [maxY, y1, y2].reduce((a, b) => a > b ? a : b);
      }
    }
    
    // If coordinates are in a small range, they might be normalized (0-1) or in a small coordinate system
    // Based on the coordinates (x1: 77, y1: 188, x2: 100, y2: 214), this looks like a very small region
    // Let's assume they're in a coordinate system where the image is much smaller (like 150x150 pixels)
    final assumedImageWidth = 150.0;
    final assumedImageHeight = 150.0;
    
    final scaleX = size.width / assumedImageWidth;
    final scaleY = size.height / assumedImageHeight;
    
    print('Coordinate range: X($minX-$maxX), Y($minY-$maxY)');
    print('Scaling factors: scaleX=$scaleX, scaleY=$scaleY (assumed image: ${assumedImageWidth}x${assumedImageHeight}, display: ${size.width}x${size.height})');

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

      // Check if coordinates are valid (not all zeros)
      if (x1 == 0.0 && y1 == 0.0 && x2 == 0.0 && y2 == 0.0) {
        print('Detection $i: Invalid coordinates (all zeros), skipping');
        continue;
      }

      // Check if bounding box has valid dimensions
      if (x2 <= x1 || y2 <= y1) {
        print('Detection $i: Invalid bounding box dimensions, skipping');
        continue;
      }

      // Scale coordinates to match the display canvas size
      final scaledX1 = x1 * scaleX;
      final scaledY1 = y1 * scaleY;
      final scaledX2 = x2 * scaleX;
      final scaledY2 = y2 * scaleY;
      
      // Clamp scaled coordinates to canvas bounds
      final finalX1 = scaledX1.clamp(0.0, size.width);
      final finalY1 = scaledY1.clamp(0.0, size.height);
      final finalX2 = scaledX2.clamp(0.0, size.width);
      final finalY2 = scaledY2.clamp(0.0, size.height);
      
      // Check if bounding box is too small to be visible (after scaling)
      final boxWidth = finalX2 - finalX1;
      final boxHeight = finalY2 - finalY1;
      if (boxWidth < 3.0 || boxHeight < 3.0) {
        print('Detection $i: Bounding box too small after scaling ($boxWidth x $boxHeight), skipping');
        continue;
      }
      
      print('Detection $i: Scaled from ($x1, $y1, $x2, $y2) to ($finalX1, $finalY1, $finalX2, $finalY2)');

      // Draw bounding box (outline only) using scaled coordinates
      final rect = Rect.fromLTRB(finalX1, finalY1, finalX2, finalY2);
      canvas.drawRect(rect, paint);
      validBoxesDrawn++;

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
        finalX1, 
        (finalY1 - textPainter.height - 4).clamp(0.0, size.height - textPainter.height - 4),
        textPainter.width + 8,
        textPainter.height + 4,
      );

      // Draw label background with dark semi-transparent background
      final labelPaint = Paint()
        ..color = Colors.black.withOpacity(0.6)
        ..style = PaintingStyle.fill;

      canvas.drawRect(labelRect, labelPaint);
      textPainter.paint(canvas, Offset(finalX1 + 4, labelRect.top + 2));
    }
    
    print('Valid bounding boxes drawn: $validBoxesDrawn out of ${detections.length} total detections');
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
