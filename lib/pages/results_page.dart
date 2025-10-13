import 'package:flutter/material.dart';
import 'dart:io';
import 'dart:math' as math;
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';
import '../widgets/bean_severity_icon.dart';

enum ResultsNavigationAction { scan, history }

class ResultsPage extends StatelessWidget {
  final BeanPrediction prediction;
  final String imagePath;
  final Map<String, dynamic>? defectDetection;
  final Map<String, dynamic>? shelfLife;

  const ResultsPage({
    super.key,
    required this.prediction,
    required this.imagePath,
    this.defectDetection,
    this.shelfLife,
  });

  @override
  Widget build(BuildContext context) {
    // Debug prints to see what data we're receiving
    print('🔍 ResultsPage Debug:');
    print('  - prediction: $prediction');
    print('  - imagePath: $imagePath');
    print('  - defectDetection: $defectDetection');
    print('  - shelfLife: $shelfLife');
    
    
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final textTheme = theme.textTheme;

    return Scaffold(
      backgroundColor: colorScheme.background,
      appBar: AppBar(
        backgroundColor: colorScheme.background,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: colorScheme.primary),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text(
          'Scanned Coffee Bean Result',
          style: textTheme.titleMedium?.copyWith(color: colorScheme.primary, fontWeight: FontWeight.w700),
        ),
        centerTitle: false,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(AppConstants.largePadding),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildImagePreview(context, colorScheme),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildInfoCard(colorScheme, textTheme),
              const SizedBox(height: AppConstants.largeSpacing),
              if (defectDetection != null) ...[
                _buildDefectDetectionCard(colorScheme),
                const SizedBox(height: AppConstants.largeSpacing),
              ],
              _buildSeverityAndDefectiveTiles(colorScheme),
              const SizedBox(height: AppConstants.largeSpacing),
              const Text(
                'Scan another image?',
                style: TextStyle(
                  color: AppColors.primaryBrown,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: AppConstants.smallSpacing),
              _buildYesNoButtons(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview(BuildContext context, ColorScheme colorScheme) {
    if (imagePath.isEmpty) {
      return Container(
        height: 220,
        width: double.infinity,
        decoration: BoxDecoration(
          color: colorScheme.surfaceVariant,
          borderRadius: BorderRadius.circular(AppConstants.largeRadius),
          border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
        ),
        clipBehavior: Clip.antiAlias,
        child: Center(
          child: Icon(Icons.image, color: colorScheme.onSurface.withOpacity(0.54), size: 48),
        ),
      );
    }

    final detections = _getDetections();
    final Size? originalSize = _resolveOriginalDetectionSize(detections);

    Widget imageStack = Stack(
      fit: StackFit.expand,
      children: [
        Positioned.fill(child: _buildImageWidget(context, colorScheme, fit: BoxFit.fill)),
        if (defectDetection != null && defectDetection!['detections'] != null)
          Positioned.fill(
            child: IgnorePointer(
              child: _buildDefectAnnotations(originalSize),
            ),
          ),
        if (defectDetection != null && defectDetection!['summary'] != null)
          _buildDefectCountOverlay(colorScheme),
      ],
    );

    if (originalSize != null && originalSize.width > 0 && originalSize.height > 0) {
      imageStack = FittedBox(
        fit: BoxFit.contain,
        alignment: Alignment.center,
        child: SizedBox(
          width: originalSize.width,
          height: originalSize.height,
          child: imageStack,
        ),
      );
    }

    return Container(
      height: 220,
      width: double.infinity,
      decoration: BoxDecoration(
        color: colorScheme.surfaceVariant,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
      ),
      clipBehavior: Clip.antiAlias,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        child: imageStack,
      ),
    );
  }

  Widget _buildImageWidget(BuildContext context, ColorScheme colorScheme, {BoxFit fit = BoxFit.cover}) {
    print('🔍 _buildImageWidget - imagePath: $imagePath');
    final isHttp = imagePath.startsWith('http');
    final isAbsolutePath = imagePath.startsWith('/') || imagePath.startsWith('http');
    final String url = imagePath.startsWith('/') ? (ApiService.apiUrl + imagePath) : imagePath;
    print('🔍 _buildImageWidget - isHttp: $isHttp, isAbsolutePath: $isAbsolutePath, url: $url');
    
    if (isHttp || imagePath.startsWith('/')) {
      return Image.network(
        url,
        fit: fit,
        width: double.infinity,
        height: double.infinity,
        errorBuilder: (c, e, s) {
          print('🔍 Image.network error: $e');
          return Center(
            child: Icon(
              Icons.broken_image,
              color: colorScheme.onSurface.withOpacity(0.54),
              size: 48,
            ),
          );
        },
        loadingBuilder: (context, child, loadingProgress) {
          if (loadingProgress == null) return child;
          return const Center(
            child: CircularProgressIndicator(),
          );
        },
      );
    }
    return Image.file(
      File(imagePath),
      fit: fit,
      width: double.infinity,
      height: double.infinity,
      errorBuilder: (c, e, s) {
        print('🔍 Image.file error: $e');
        return Center(
          child: Icon(
            Icons.broken_image,
            color: colorScheme.onSurface.withOpacity(0.54),
            size: 48,
          ),
        );
      },
    );
  }

  Widget _buildDefectAnnotations(Size? originalSize) {
    final detections = _getDetections();
    if (detections.isEmpty) {
      return const SizedBox.shrink();
    }

    // Debug: Print detection data
    print('Defect detections: ${detections.length}');
    for (int i = 0; i < detections.length; i++) {
      print('Detection $i: ${detections[i]}');
    }

    return CustomPaint(
      painter: DefectAnnotationPainter(
        detections,
        originalSize: originalSize,
      ),
    );
  }

  Size? _resolveOriginalDetectionSize(List<dynamic> detections) {
    for (final detection in detections) {
      if (detection is Map<String, dynamic>) {
        if (detection['image_width'] != null && detection['image_height'] != null) {
          final width = (detection['image_width'] as num).toDouble();
          final height = (detection['image_height'] as num).toDouble();
          if (width > 0 && height > 0) {
            return Size(width, height);
          }
        }
        if (detection['image_size'] is Map) {
          final dims = Map<String, dynamic>.from(detection['image_size'] as Map);
          final width = (dims['width'] as num?)?.toDouble();
          final height = (dims['height'] as num?)?.toDouble();
          if (width != null && height != null && width > 0 && height > 0) {
            return Size(width, height);
          }
        }
      }
    }
    return null;
  }

  Widget _buildDefectCountOverlay(ColorScheme colorScheme) {
    final summary = _getDefectSummary();
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

  Widget _buildInfoCard(ColorScheme colorScheme, TextTheme textTheme) {
    final DateTime now = DateTime.now();
    final String dateStr = '${now.month}/${now.day}/${now.year} - ${now.hour}:${now.minute.toString().padLeft(2, '0')}';
    final double healthyPct = (prediction.confidence * 100).clamp(0.0, 100.0);
    final Map<String, dynamic>? shelfLifeData = shelfLife != null
        ? Map<String, dynamic>.from(shelfLife!)
        : null;
    final double? estimatedMonths = (shelfLifeData?['estimated_months'] as num?)?.toDouble();
    final Map<String, dynamic>? monthsRange = shelfLifeData?['estimated_months_range'] is Map
        ? Map<String, dynamic>.from(shelfLifeData!['estimated_months_range'] as Map)
        : null;
    final Map<String, dynamic>? defectSummary = _getDefectSummary();
    final String? severityLabel = (shelfLifeData?['severity'] as String?) ?? (defectSummary?['severity'] as String?);
    final double confidenceScore = shelfLifeData != null
        ? (((shelfLifeData['confidence_score'] ?? shelfLifeData['confidence'] ?? 0.0) as num?)?.toDouble() ?? 0.0)
        : (healthyPct / 100.0);
    final String statusLabel = _resolveStatus(
      category: shelfLifeData?['category'] as String?,
      severity: severityLabel,
      predictedDays: shelfLifeData?['predicted_days'] as num?,
    );
    final onSurface = colorScheme.onSurface;
    final surface = colorScheme.surface;
    final chipBackground = _getShelfLifeColor(
      colorScheme,
      (shelfLife?['category'] as String?) ?? _deriveShelfLifeCategory(shelfLife ?? {}),
    );

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  dateStr,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: onSurface.withOpacity(0.7)),
                ),
              ),
              IconButton(
                onPressed: () {},
                icon: Icon(Icons.download, size: 18, color: onSurface.withOpacity(0.7)),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          Row(
            children: [
              Text(
                'Type: ',
                style: TextStyle(fontWeight: FontWeight.w600, color: onSurface.withOpacity(0.75)),
              ),
              Text(
                prediction.prediction,
                style: TextStyle(color: onSurface),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Divider(
            color: AppColors.dividerGrey,
            height: 24,
          ),
          Text(
            'Estimated Shelf Life',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: onSurface.withOpacity(0.8),
            ),
          ),
          const SizedBox(height: 8),

          // Shelf Life Days
          if (shelfLife != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Predicted Days:',
                  style: TextStyle(fontWeight: FontWeight.w500, color: onSurface.withOpacity(0.75)),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: chipBackground,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Text(
                    '${shelfLife!['predicted_days'] ?? 0} days',
                    style: TextStyle(
                      color: chipBackground.computeLuminance() > 0.5 ? Colors.black87 : Colors.white,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
          if (estimatedMonths != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
               children: [
                 const Text('Estimated Months:', style: TextStyle(fontWeight: FontWeight.w500, color: AppColors.textDarkGrey)),
                 Text(
                   _formatMonthsText(estimatedMonths, monthsRange),
                   style: const TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey),
                 ),
               ],
            ),
            const SizedBox(height: 8),
          ],
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Status:', style: TextStyle(fontWeight: FontWeight.w500, color: AppColors.textDarkGrey)),
              Text(
                statusLabel,
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: _getShelfLifeTextColor(colorScheme, statusLabel),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
        ] else
          const SizedBox(height: AppConstants.mediumSpacing),
          
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Confidence Score:', style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey)),
              Text(
                '${(confidenceScore * 100).clamp(0, 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDarkGrey,
                  fontSize: 16,
                ),
               ),
             ],
           ),
        ],
      ),
    );
  }

  Widget _buildDefectDetectionCard(ColorScheme colorScheme) {
    if (defectDetection == null) return const SizedBox.shrink();
    
    final summary = _getDefectSummary() ?? <String, dynamic>{};
    final detections = _getDetections();
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
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
          const SizedBox(height: 12),
          
          // Total Defects
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Total Defects:', style: TextStyle(fontWeight: FontWeight.w500, color: AppColors.textDarkGrey)),
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
          
          // Defect Types
          if (summary['defect_types'] != null && (summary['defect_types'] as Map).isNotEmpty) ...[
            const Text('Defect Types:', style: TextStyle(
              fontWeight: FontWeight.w500,
              color: AppColors.textDarkGrey,
            )),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 4,
              children: (Map<String, dynamic>.from(summary['defect_types'] as Map)).entries
                  .map((entry) => Chip(
                        label: Text('${entry.key}: ${entry.value}'),
                        backgroundColor: colorScheme.secondary.withOpacity(0.2),
                        labelStyle: const TextStyle(fontSize: 12),
                      ))
                  .toList(),
            ),
            const SizedBox(height: 12),
          ],
          if (detections.isNotEmpty) ...[
            const Text('Detected Areas:', style: TextStyle(
              fontWeight: FontWeight.w500,
              color: AppColors.textDarkGrey,
            )),
            const SizedBox(height: 8),
            ...List.generate(detections.length, (index) {
              final raw = detections[index];
              if (raw is Map<String, dynamic>) {
                return _defectDetailRow(raw, index);
              }
              return _defectDetailRow(Map<String, dynamic>.from(raw as Map), index);
            }),
          ],
        ],
      ),
    );
  }

  Color _getShelfLifeColor(ColorScheme colorScheme, String? category) {
    switch (category?.toLowerCase()) {
      case 'excellent':
        return Colors.green;
      case 'good':
        return Colors.blue;
      case 'warning':
        return colorScheme.secondary;
      case 'critical':
        return Colors.red;
      case 'expired':
        return Colors.red.shade800;
      default:
        return Colors.grey;
    }
  }

  Color _getShelfLifeTextColor(ColorScheme colorScheme, String? category) {
    switch (category?.toLowerCase()) {
      case 'excellent':
        return Colors.green;
      case 'good':
        return Colors.blue;
      case 'warning':
        return colorScheme.secondary;
      case 'critical':
        return Colors.red;
      case 'expired':
        return Colors.red.shade800;
      default:
        return Colors.grey;
    }
  }

  String _formatSeverityLabel(String? severity) {
    if (severity == null || severity.isEmpty) return 'Unknown';
    final lower = severity.toLowerCase();
    switch (lower) {
      case 'mild':
        return 'Mild';
      case 'moderate':
        return 'Moderate';
      case 'severe':
        return 'Severe';
      default:
        return lower[0].toUpperCase() + lower.substring(1);
    }
  }

  String _computeSeverityFromPercentage(double percentage) {
    if (percentage < 22) return 'mild';
    if (percentage < 78) return 'moderate';
    return 'severe';
  }

  String _formatDefectType(String raw) {
    if (raw.isEmpty) return 'Unknown';
    final normalized = raw.replaceAll('_', ' ').replaceAll('-', ' ');
    return normalized
        .split(' ')
        .where((part) => part.isNotEmpty)
        .map((part) => part[0].toUpperCase() + part.substring(1))
        .join(' ');
  }

  Widget _defectDetailRow(Map<String, dynamic> detection, int index) {
    final type = _formatDefectType((detection['defect_type'] as String?) ?? 'Unknown');
    final coords = detection['coordinates'] as Map<String, dynamic>?;
    final position = coords != null
        ? '(${(coords['x1'] as num?)?.toInt() ?? 0}, ${(coords['y1'] as num?)?.toInt() ?? 0})'
        : '';

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 10),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              color: AppColors.primaryBrown.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            alignment: Alignment.center,
            child: Text(
              '${index + 1}',
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                color: AppColors.primaryBrown,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  type,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    color: AppColors.textDarkGrey,
                  ),
                ),
                if (position.isNotEmpty)
                  Text(
                    'Location: $position',
                    style: const TextStyle(fontSize: 12, color: AppColors.textGrey),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatMonthsText(double estimate, Map<String, dynamic>? range) {
    final double? min = (range?['min'] as num?)?.toDouble();
    final double? max = (range?['max'] as num?)?.toDouble();
    final estimateLabel = estimate.toStringAsFixed(1);
    if (min != null && max != null && min > 0 && max > 0) {
      return '$estimateLabel mo (${min.toStringAsFixed(1)}-${max.toStringAsFixed(1)} mo)';
    }
    return '$estimateLabel mo';
  }

  String _resolveStatus({String? category, String? severity, num? predictedDays}) {
    String normalizedSeverity = severity?.toLowerCase() ?? '';
    if (normalizedSeverity.isEmpty && category != null) {
      switch (category.toLowerCase()) {
        case 'excellent':
          normalizedSeverity = 'mild';
          break;
        case 'good':
        case 'warning':
          normalizedSeverity = 'moderate';
          break;
        case 'critical':
        case 'expired':
          normalizedSeverity = 'severe';
          break;
      }
    }

    if (normalizedSeverity.isEmpty && predictedDays != null) {
      normalizedSeverity = _computeSeverityFromPercentage(
        predictedDays.toDouble() > 0 ? (predictedDays / 240.0) * 100.0 : 0,
      );
    }

    String severityStatus;
    switch (normalizedSeverity) {
      case 'mild':
        severityStatus = 'Excellent';
        break;
      case 'moderate':
        severityStatus = 'Warning';
        break;
      case 'severe':
        severityStatus = 'Critical';
        break;
      default:
        severityStatus = category ?? 'Unknown';
    }

    if (category != null) {
      final normalizedCategory = category.toLowerCase();
      if ((normalizedSeverity == 'moderate' && normalizedCategory == 'good') ||
          (normalizedSeverity == 'severe' && normalizedCategory == 'warning')) {
        severityStatus = _capitalize(normalizedCategory);
      } else if (normalizedCategory == 'excellent' ||
          normalizedCategory == 'good' ||
          normalizedCategory == 'warning' ||
          normalizedCategory == 'critical' ||
          normalizedCategory == 'expired') {
        severityStatus = _capitalize(normalizedCategory);
      }
    }

    return severityStatus;
  }

  String _capitalize(String value) {
    if (value.isEmpty) return value;
    return value[0].toUpperCase() + value.substring(1).toLowerCase();
  }


  
  Widget _buildSeverityAndDefectiveTiles(ColorScheme colorScheme) {
    // Derive defective% from summary if available; else from (1 - confidence)
    double defectivePct = 0;
    final summary = _getDefectSummary();
    print('=== DEFECTIVE PERCENTAGE DEBUG ===');
    print('Summary data: $summary');
    print('DefectDetection data: $defectDetection');
    
    String? severityLabel;
    if (summary != null) {
      print('Summary keys: ${summary.keys}');
      print('defect_percentage: ${summary['defect_percentage']}');
      print('total_defects: ${summary['total_defects']}');
      severityLabel = (summary['severity'] as String?);
      
      final num? rawDefectPercentage = summary['defect_percentage'] as num?;
      if (rawDefectPercentage != null) {
        defectivePct = rawDefectPercentage.toDouble();
        print('Defective percentage from summary defect_percentage: $defectivePct');
      } else if (summary['total_defects'] != null && (summary['total_defects'] as num).toInt() > 0) {
        final totalDefects = (summary['total_defects'] as num).toInt();
        final estimatedTotalBeans = 15;
        defectivePct = (totalDefects / estimatedTotalBeans) * 100.0;
        print('Defective percentage calculated from total_defects: $totalDefects out of $estimatedTotalBeans = $defectivePct%');
      } else {
        defectivePct = 0.0;
        print('No valid defect data available in summary');
      }
    } else {
      defectivePct = (1.0 - prediction.confidence) * 100.0;
      print('Defective percentage from confidence: $defectivePct (confidence: ${prediction.confidence})');
    }
    defectivePct = defectivePct.clamp(0.0, 100.0);
    print('Final defective percentage: $defectivePct');
    print('=== END DEBUG ===');

    int severityLevel;
    if (severityLabel != null) {
      switch (severityLabel.toLowerCase()) {
        case 'mild':
          severityLevel = 1;
          break;
        case 'moderate':
          severityLevel = 2;
          break;
        case 'severe':
          severityLevel = 3;
          break;
        default:
          severityLevel = defectivePct < 15 ? 1 : (defectivePct < 35 ? 2 : 3);
      }
    } else {
      severityLevel = defectivePct < 15 ? 1 : (defectivePct < 35 ? 2 : 3);
    }

    return Row(
      children: [
        Expanded(
          child: _severityCard(
            colorScheme,
            severityLevel,
            qualityGrade: summary?['quality_grade'] as String?,
            severityLabel: severityLabel,
          ),
        ),
        const SizedBox(width: AppConstants.mediumSpacing),
        Expanded(
          child: _defectivePercentCard(colorScheme, defectivePct),
        ),
      ],
    );
  }

  // ---------- Normalization helpers for history payloads ----------
  Map<String, dynamic>? _getDefectSummary() {
    if (defectDetection == null) return null;
    Map<String, dynamic> summary = {};
    if (defectDetection!['summary'] is Map<String, dynamic>) {
      summary = Map<String, dynamic>.from(defectDetection!['summary'] as Map);
    } else {
      final dd = Map<String, dynamic>.from(defectDetection!);
      final String type = (dd['defect_type'] as String?) ?? 'unknown';
      final num percentage = (dd['defect_percentage'] as num?) ?? 0.0;
      final String grade = _deriveQualityGrade(percentage.toDouble());
      summary = {
        'quality_grade': grade,
        'total_defects': type == 'unknown' ? 0 : 1,
        'defect_types': type == 'unknown' ? {} : {type: 1},
        'defect_percentage': percentage.toDouble(),
      };
    }

    final detections = _getDetections();
    if (!summary.containsKey('total_defects') ||
        summary['total_defects'] == null ||
        (summary['total_defects'] as num).toInt() == 0 && detections.isNotEmpty) {
      summary['total_defects'] = detections.length;
    }

    final Map<String, int> typeCounts = {};
    for (final detection in detections) {
      final rawType = (detection['defect_type'] as String? ?? 'Unknown').toLowerCase();
      typeCounts[rawType] = (typeCounts[rawType] ?? 0) + 1;
    }
    if (typeCounts.isNotEmpty) {
      summary['defect_types'] = typeCounts.map((key, value) => MapEntry(_formatDefectType(key), value));
    }

    if (!summary.containsKey('severity') || summary['severity'] == null) {
      final double pct = (summary['defect_percentage'] as num?)?.toDouble() ?? 0.0;
      summary['severity'] = _computeSeverityFromPercentage(pct);
    }

    if (!summary.containsKey('defect_percentage') ||
        (summary['defect_percentage'] as num?) == null) {
      final severity = summary['severity'] as String?;
      double fallbackPercentage;
      switch (severity) {
        case 'mild':
          fallbackPercentage = 8.0;
          break;
        case 'moderate':
          fallbackPercentage = 45.0;
          break;
        case 'severe':
          fallbackPercentage = 85.0;
          break;
        default:
          fallbackPercentage = detections.length * 12.0;
      }
      summary['defect_percentage'] = fallbackPercentage.clamp(0.0, 100.0);
    }

    return summary;
  }

  List<dynamic> _getDetections() {
    if (defectDetection == null) return const [];
    if (defectDetection!['detections'] is List) {
      return List<dynamic>.from(defectDetection!['detections'] as List);
    }
    final dd = Map<String, dynamic>.from(defectDetection!);
    final coords = Map<String, dynamic>.from(
      (dd['defect_coordinates'] as Map?) ?? <String, dynamic>{}
    );
    return [
      {
        'defect_type': dd['defect_type'] ?? 'unknown',
        'confidence': (dd['confidence'] as num?)?.toDouble() ?? 0.0,
        'coordinates': {
          'x1': (coords['x1'] as num?)?.toDouble() ?? 0.0,
          'y1': (coords['y1'] as num?)?.toDouble() ?? 0.0,
          'x2': (coords['x2'] as num?)?.toDouble() ?? 0.0,
          'y2': (coords['y2'] as num?)?.toDouble() ?? 0.0,
        },
      }
    ];
  }

  // Derive a readable shelf-life status if backend did not store the category
  String _deriveShelfLifeCategory(Map<String, dynamic> shelf) {
    final int days = (shelf['predicted_days'] as num?)?.toInt() ?? 0;
    if (days >= 30) return 'Excellent';
    if (days >= 20) return 'Good';
    if (days >= 10) return 'Warning';
    if (days > 0) return 'Critical';
    return 'Unknown';
  }

  // Derive a quality grade from defect percentage for history rows
  String _deriveQualityGrade(double defectivePct) {
    if (defectivePct < 10) return 'A';
    if (defectivePct < 20) return 'B';
    if (defectivePct < 35) return 'C';
    if (defectivePct < 50) return 'D';
    return 'F';
  }

  Widget _severityCard(ColorScheme colorScheme, int severityLevel, {String? qualityGrade, String? severityLabel}) {
    final computedLabel = severityLabel != null && severityLabel.isNotEmpty
        ? _formatSeverityLabel(severityLabel)
        : (severityLevel == 1 ? 'Mild' : (severityLevel == 2 ? 'Moderate' : 'Severe'));
    return Container(
      padding: const EdgeInsets.all(AppConstants.largePadding),
      constraints: const BoxConstraints(minHeight: 220),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
         children: [
           const Text('Severity:', style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey)),
           const SizedBox(height: 8),
          Center(child: BeanSeverityIcon(severityLevel: severityLevel, size: 72, color: AppColors.primaryBrown)),
          const SizedBox(height: AppConstants.smallSpacing),
         Center(
           child: Text(
             computedLabel,
              style: const TextStyle(color: AppColors.textDarkGrey, fontWeight: FontWeight.w600),
           ),
         ),
         if (qualityGrade != null) ...[
           const SizedBox(height: 6),
           Center(
             child: Text(
               'Quality: $qualityGrade',
               style: const TextStyle(color: AppColors.textGrey, fontSize: 12, fontWeight: FontWeight.w500),
             ),
           ),
         ],
        ],
      ),
    );
  }

  Widget _defectivePercentCard(ColorScheme colorScheme, double percent) {
    percent = percent.clamp(0, 100);
    return Container(
      padding: const EdgeInsets.all(AppConstants.largePadding),
      constraints: const BoxConstraints(minHeight: 220),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
           const Text('Defective (%)', style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey)),
           const SizedBox(height: 16), // Increased spacing
          Center(child: _circularPercent(colorScheme, percent: percent, color: AppColors.primaryBrown)),
          const SizedBox(height: 16), // Added bottom spacing
        ],
      ),
    );
  }

   Widget _circularPercent(ColorScheme colorScheme, {required double percent, required Color color}) {
     print('Circular percent widget - percent: $percent, color: $color');
     return SizedBox(
       height: 120,
       width: 120,
       child: Stack(
         alignment: Alignment.center,
         children: [
           SizedBox(
             height: 110,
             width: 110,
             child: CircularProgressIndicator(
               value: (percent / 100.0).clamp(0.0, 1.0),
               strokeWidth: 8,
               backgroundColor: colorScheme.surfaceVariant,
               valueColor: AlwaysStoppedAnimation<Color>(color),
             ),
           ),
           Container(
             width: 72,
             height: 72,
             decoration: BoxDecoration(
               color: colorScheme.surface,
               borderRadius: BorderRadius.circular(36),
               boxShadow: [
                 BoxShadow(
                   color: Colors.black12,
                   blurRadius: 4,
                 ),
               ],
             ),
             alignment: Alignment.center,
             child: Text(
               '${percent.toStringAsFixed(0)}%',
               style: const TextStyle(
                 fontWeight: FontWeight.w700,
                 fontSize: 18,
                 color: AppColors.primaryBrown,
               ),
             ),
           ),
         ],
       ),
     );
   }

  Widget _buildYesNoButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).pop(ResultsNavigationAction.scan),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryBrown,
              foregroundColor: Theme.of(context).colorScheme.onSurface,
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
            onPressed: () => Navigator.of(context).pop(ResultsNavigationAction.history),
            style: ElevatedButton.styleFrom(
              backgroundColor: Theme.of(context).colorScheme.surfaceTint,
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
  final Size? originalSize;
  
  DefectAnnotationPainter(this.detections, {this.originalSize});

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
    
    double minX = double.infinity;
    double minY = double.infinity;
    for (final detection in detections) {
      final coords = detection['coordinates'] as Map<String, dynamic>?;
      if (coords == null) continue;
      final x1 = (coords['x1'] as num?)?.toDouble() ?? 0.0;
      final y1 = (coords['y1'] as num?)?.toDouble() ?? 0.0;
      final x2 = (coords['x2'] as num?)?.toDouble() ?? 0.0;
      final y2 = (coords['y2'] as num?)?.toDouble() ?? 0.0;
      minX = math.min(minX, math.min(x1, x2));
      minY = math.min(minY, math.min(y1, y2));
    }
    if (!minX.isFinite) minX = 0;
    if (!minY.isFinite) minY = 0;

    final bool hasOriginalSize = originalSize != null && originalSize!.width > 0 && originalSize!.height > 0;
    if (hasOriginalSize) {
      minX = 0;
      minY = 0;
    }

    final Size sourceSize = hasOriginalSize
        ? originalSize!
        : _calculateDetectionBounds(detections, minX: minX, minY: minY);
    final double scaleX = size.width / (sourceSize.width == 0 ? 1 : sourceSize.width);
    final double scaleY = size.height / (sourceSize.height == 0 ? 1 : sourceSize.height);
    
    print('Detected source size: $sourceSize -> scaleX=$scaleX, scaleY=$scaleY');

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
      if ((x1 == 0.0 && y1 == 0.0 && x2 == 0.0 && y2 == 0.0) || x2 <= x1 || y2 <= y1) {
        print('Detection $i: Invalid coordinates, skipping visual overlay but keeping in counts.');
        continue;
      }

      // Scale coordinates to match the display canvas size
      final scaledX1 = (x1 - minX) * scaleX;
      final scaledY1 = (y1 - minY) * scaleY;
      final scaledX2 = (x2 - minX) * scaleX;
      final scaledY2 = (y2 - minY) * scaleY;
      
      // Clamp scaled coordinates to canvas bounds
      final finalX1 = scaledX1.clamp(0.0, size.width);
      final finalY1 = scaledY1.clamp(0.0, size.height);
      final finalX2 = scaledX2.clamp(0.0, size.width);
      final finalY2 = scaledY2.clamp(0.0, size.height);
      
      // Check if bounding box is too small to be visible (after scaling)
      final boxWidth = finalX2 - finalX1;
      final boxHeight = finalY2 - finalY1;
      if (boxWidth < 1.5 || boxHeight < 1.5) {
        print('Detection $i: Bounding box very small after scaling ($boxWidth x $boxHeight), drawing indicator dot instead');
        final dotPaint = Paint()
          ..color = paint.color
          ..style = PaintingStyle.fill;
        final dotCenter = Offset(finalX1.clamp(4.0, size.width - 4.0), finalY1.clamp(4.0, size.height - 4.0));
        canvas.drawCircle(dotCenter, 4, dotPaint);

        // Draw numeric badge near the detection
        const badgeTextStyle = TextStyle(
          color: Colors.white,
          fontSize: 10,
          fontWeight: FontWeight.w700,
        );
        textPainter.text = TextSpan(
          text: '${i + 1}',
          style: badgeTextStyle,
        );
        textPainter.layout();
        const badgePadding = 4.0;
        final badgeWidth = textPainter.width + badgePadding * 2;
        final badgeHeight = textPainter.height + badgePadding;
        final badgeLeft = (dotCenter.dx - badgeWidth / 2).clamp(0.0, size.width - badgeWidth);
        final badgeTop = (dotCenter.dy + 6).clamp(0.0, size.height - badgeHeight);
        final badgeRect = Rect.fromLTWH(badgeLeft, badgeTop, badgeWidth, badgeHeight);
        final badgePaint = Paint()
          ..color = Colors.redAccent.withOpacity(0.95)
          ..style = PaintingStyle.fill;
        canvas.drawRRect(RRect.fromRectAndRadius(badgeRect, const Radius.circular(6)), badgePaint);
        textPainter.paint(canvas, Offset(badgeRect.left + badgePadding, badgeRect.top + badgePadding / 2));

        final labelText = '${i + 1}. $defectType (${(confidence * 100).toInt()}%)';
        textPainter.text = TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.bold,
          ),
        );
        textPainter.layout();
        final labelRect = Rect.fromLTWH(
          dotCenter.dx,
          (dotCenter.dy - textPainter.height - 4).clamp(0.0, size.height - textPainter.height - 4),
          textPainter.width + 8,
          textPainter.height + 4,
        );
        final labelPaint = Paint()
          ..color = Colors.black.withOpacity(0.7)
          ..style = PaintingStyle.fill;
        canvas.drawRect(labelRect, labelPaint);
        textPainter.paint(canvas, Offset(labelRect.left + 4, labelRect.top + 2));

        validBoxesDrawn++;
        continue;
      }
      
      print('Detection $i: Scaled from ($x1, $y1, $x2, $y2) to ($finalX1, $finalY1, $finalX2, $finalY2)');

      // Draw bounding box (outline only) using scaled coordinates
      final rect = Rect.fromLTRB(finalX1, finalY1, finalX2, finalY2);
      canvas.drawRect(rect, paint);
      validBoxesDrawn++;

      // Draw numeric badge anchored to the top-left corner of the bounding box
      const badgeTextStyle = TextStyle(
        color: Colors.white,
        fontSize: 11,
        fontWeight: FontWeight.w700,
      );
      textPainter.text = TextSpan(
        text: '${i + 1}',
        style: badgeTextStyle,
      );
      textPainter.layout();
      const badgePadding = 4.0;
      final badgeWidth = textPainter.width + badgePadding * 2;
      final badgeHeight = textPainter.height + badgePadding;
      final badgeLeft = (finalX1 + 2).clamp(0.0, size.width - badgeWidth);
      final badgeTop = (finalY1 + 2).clamp(0.0, size.height - badgeHeight);
      final badgeRect = Rect.fromLTWH(badgeLeft, badgeTop, badgeWidth, badgeHeight);
      final badgePaint = Paint()
        ..color = Colors.redAccent.withOpacity(0.95)
        ..style = PaintingStyle.fill;
      canvas.drawRRect(RRect.fromRectAndRadius(badgeRect, const Radius.circular(6)), badgePaint);
      textPainter.paint(canvas, Offset(badgeRect.left + badgePadding, badgeRect.top + badgePadding / 2));

      // Re-layout with descriptive label that mirrors the results list numbering
      final labelText = '${i + 1}. $defectType (${(confidence * 100).toInt()}%)';
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

  static Size _calculateDetectionBounds(List<dynamic> detections, {double minX = 0, double minY = 0}) {
    double maxX = 0;
    double maxY = 0;
    for (final detection in detections) {
      final coords = detection['coordinates'] as Map<String, dynamic>?;
      if (coords == null) continue;
      final x1 = (coords['x1'] as num?)?.toDouble() ?? 0.0;
      final y1 = (coords['y1'] as num?)?.toDouble() ?? 0.0;
      final x2 = (coords['x2'] as num?)?.toDouble() ?? 0.0;
      final y2 = (coords['y2'] as num?)?.toDouble() ?? 0.0;
      maxX = [maxX, x1, x2].reduce((a, b) => a > b ? a : b);
      maxY = [maxY, y1, y2].reduce((a, b) => a > b ? a : b);
    }
    maxX = math.max(0, maxX - minX);
    maxY = math.max(0, maxY - minY);
    if (maxX <= 1.0 && maxY <= 1.0) {
      // normalized coordinates (0-1). treat as normalized square.
      return const Size(1, 1);
    }
    return Size(maxX, maxY);
  }
}
