import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_file_dialog/flutter_file_dialog.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:permission_handler/permission_handler.dart';
import 'package:share_plus/share_plus.dart';

import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';
import '../utils/app_logger.dart';
import '../widgets/bean_severity_icon.dart';

enum ResultsNavigationAction { scan, history }

void _logResultsPage(
  String message, {
  Object? error,
  StackTrace? stackTrace,
}) {
  logDebug(
    'ResultsPage',
    message,
    error: error,
    stackTrace: stackTrace,
  );
}

class ResultsPage extends StatelessWidget {
  static final Expando<bool> _autoSaveGuard = Expando<bool>();
  final BeanPrediction prediction;
  final String imagePath;
  final Map<String, dynamic>? defectDetection;
  final Map<String, dynamic>? shelfLife;
  final bool shouldAutoSave;
  final bool fromHistory;

  const ResultsPage({
    super.key,
    required this.prediction,
    required this.imagePath,
    this.defectDetection,
    this.shelfLife,
    this.shouldAutoSave = false,
    this.fromHistory = false,
  });

  @override
  Widget build(BuildContext context) {
    // Debug log to see what data we're receiving
    _logResultsPage(
      'ResultsPage Debug:\n'
      '  - prediction: $prediction\n'
      '  - imagePath: $imagePath\n'
      '  - defectDetection: $defectDetection\n'
      '  - shelfLife: $shelfLife',
    );

    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final textTheme = theme.textTheme;
    final scaffoldBackground = theme.scaffoldBackgroundColor;

    if (shouldAutoSave && (_autoSaveGuard[this] != true)) {
      _autoSaveGuard[this] = true;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _downloadResultPdf(context, silent: true);
      });
    }

    return Scaffold(
      backgroundColor: scaffoldBackground,
      appBar: AppBar(
        backgroundColor: scaffoldBackground,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: colorScheme.primary),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text(
          fromHistory ? 'Coffee Bean History Record' : 'Scanned Coffee Bean Result',
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
              _buildInfoCard(context, colorScheme, textTheme),
              const SizedBox(height: AppConstants.largeSpacing),
              if (defectDetection != null) ...[
                _buildDefectDetectionCard(colorScheme),
                const SizedBox(height: AppConstants.largeSpacing),
              ],
              _buildSeverityAndDefectiveTiles(colorScheme),
              const SizedBox(height: AppConstants.largeSpacing),
              if (!fromHistory) ...[
                Text(
                  'Scan another image?',
                  style: textTheme.titleSmall?.copyWith(
                    color: colorScheme.primary,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: AppConstants.smallSpacing),
                _buildYesNoButtons(context),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _downloadResultPdf(BuildContext context, {bool silent = false}) async {
    final messenger = ScaffoldMessenger.maybeOf(context);

    try {
      final now = DateTime.now();
      final dateLabel =
          '${now.month.toString().padLeft(2, '0')}/${now.day.toString().padLeft(2, '0')}/${now.year} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';
      final shelfLifeData =
          shelfLife != null ? Map<String, dynamic>.from(shelfLife!) : null;
      final defectSummary = _getDefectSummary();
      final detections = _getDetections();
      final probabilityEntries = prediction.allProbabilities.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));

      final doc = pw.Document();
      final sectionTitleStyle =
          pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold);
      final bodyStyle = pw.TextStyle(fontSize: 12, color: PdfColors.grey800);
      final accentStyle = bodyStyle.copyWith(fontWeight: pw.FontWeight.bold);
      final tableHeaderStyle = pw.TextStyle(
        fontSize: 11,
        fontWeight: pw.FontWeight.bold,
        color: PdfColors.white,
      );

      doc.addPage(
        pw.MultiPage(
          pageTheme: pw.PageTheme(
            margin: const pw.EdgeInsets.all(32),
          ),
          build: (pw.Context _) {
            final widgets = <pw.Widget>[
              pw.Text(
                'BeanScan Analysis Report',
                style: pw.TextStyle(fontSize: 22, fontWeight: pw.FontWeight.bold),
              ),
              pw.SizedBox(height: 6),
              pw.Text(
                'Generated on $dateLabel',
                style: bodyStyle.copyWith(color: PdfColors.grey600),
              ),
              pw.SizedBox(height: 20),
              pw.Text('Classification', style: sectionTitleStyle),
              pw.SizedBox(height: 6),
              pw.Bullet(
                text: 'Predicted type: ${prediction.prediction}',
                style: bodyStyle,
              ),
              pw.Bullet(
                text: 'Confidence: ${(prediction.confidence * 100).toStringAsFixed(1)}%',
                style: bodyStyle,
              ),
            ];

            if (probabilityEntries.isNotEmpty) {
              widgets.add(pw.SizedBox(height: 10));
              widgets.add(
                pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.Text('Class probabilities', style: accentStyle),
                    pw.SizedBox(height: 4),
                    pw.Table(
                      border: pw.TableBorder.symmetric(
                        inside: const pw.BorderSide(
                          color: PdfColors.grey400,
                          width: 0.3,
                        ),
                        outside: const pw.BorderSide(
                          color: PdfColors.grey400,
                          width: 0.5,
                        ),
                      ),
                      columnWidths: const {
                        0: pw.FlexColumnWidth(2),
                        1: pw.FlexColumnWidth(1),
                      },
                      children: [
                        pw.TableRow(
                          decoration: const pw.BoxDecoration(
                            color: PdfColors.blueGrey700,
                          ),
                          children: [
                            pw.Padding(
                              padding: const pw.EdgeInsets.symmetric(
                                vertical: 4,
                                horizontal: 6,
                              ),
                              child: pw.Text('Class', style: tableHeaderStyle),
                            ),
                            pw.Padding(
                              padding: const pw.EdgeInsets.symmetric(
                                vertical: 4,
                                horizontal: 6,
                              ),
                              child:
                                  pw.Text('Confidence', style: tableHeaderStyle),
                            ),
                          ],
                        ),
                        ...probabilityEntries.map(
                          (entry) => pw.TableRow(
                            children: [
                              pw.Padding(
                                padding: const pw.EdgeInsets.symmetric(
                                  vertical: 4,
                                  horizontal: 6,
                                ),
                                child: pw.Text(
                                  _formatDefectType(entry.key),
                                  style: bodyStyle,
                                ),
                              ),
                              pw.Padding(
                                padding: const pw.EdgeInsets.symmetric(
                                  vertical: 4,
                                  horizontal: 6,
                                ),
                                child: pw.Text(
                                  '${(entry.value * 100).toStringAsFixed(1)}%',
                                  style: bodyStyle,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            }

            if (shelfLifeData != null && shelfLifeData.isNotEmpty) {
              final double? predictedDays =
                  (shelfLifeData['predicted_days'] as num?)?.toDouble();
              final double? estimatedMonths =
                  (shelfLifeData['estimated_months'] as num?)?.toDouble();
              final Map<String, dynamic>? monthsRange =
                  shelfLifeData['estimated_months_range'] is Map
                      ? Map<String, dynamic>.from(
                          shelfLifeData['estimated_months_range'] as Map,
                        )
                      : null;
              final double? shelfConfidence =
                  (shelfLifeData['confidence_score'] ??
                          shelfLifeData['confidence']) is num
                      ? ((shelfLifeData['confidence_score'] ??
                              shelfLifeData['confidence']) as num)
                          .toDouble()
                      : null;
              final String status = _resolveStatus(
                category: shelfLifeData['category'] as String?,
                severity: (shelfLifeData['severity'] as String?) ??
                    (defectSummary?['severity'] as String?),
                predictedDays: shelfLifeData['predicted_days'] as num?,
              );

              widgets.add(pw.SizedBox(height: 18));
              widgets.add(pw.Text('Shelf life', style: sectionTitleStyle));
              widgets.add(pw.SizedBox(height: 6));
              final shelfLines = <String>[
                if (predictedDays != null)
                  'Predicted days: ${predictedDays.toStringAsFixed(0)}',
                if (estimatedMonths != null)
                  'Estimated months: ${_formatMonthsText(estimatedMonths, monthsRange)}',
                if (shelfConfidence != null)
                  'Model confidence: ${(shelfConfidence * 100).clamp(0, 100).toStringAsFixed(1)}%',
                if (shelfLifeData['severity'] != null)
                  'Severity: ${_formatDefectType((shelfLifeData['severity'] as String?) ?? '')}',
                if (status.isNotEmpty) 'Status: $status',
              ];
              widgets.add(
                pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: shelfLines
                      .where((line) => line.isNotEmpty)
                      .map((line) => pw.Text(line, style: bodyStyle))
                      .toList(),
                ),
              );
            }

            if (defectSummary != null && defectSummary.isNotEmpty) {
              final Map<String, int> defectTypes = {};
              final rawTypes = defectSummary['defect_types'];
              if (rawTypes is Map) {
                rawTypes.forEach((key, value) {
                  if (value is num) {
                    defectTypes[key.toString()] = value.toInt();
                  }
                });
              }

              widgets.add(pw.SizedBox(height: 18));
              widgets.add(pw.Text('Defect summary', style: sectionTitleStyle));
              widgets.add(pw.SizedBox(height: 6));
              widgets.add(
                pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    if (defectSummary['quality_grade'] != null)
                      pw.Text(
                        'Quality grade: ${defectSummary['quality_grade']}',
                        style: bodyStyle,
                      ),
                    if (defectSummary['total_defects'] != null)
                      pw.Text(
                        'Total defects detected: ${(defectSummary['total_defects'] as num).toInt()}',
                        style: bodyStyle,
                      ),
                    if (defectSummary['defect_percentage'] != null)
                      pw.Text(
                        'Defect percentage: ${((defectSummary['defect_percentage'] as num?)?.toDouble() ?? 0).toStringAsFixed(1)}%',
                        style: bodyStyle,
                      ),
                    if (defectSummary['severity'] != null)
                      pw.Text(
                        'Severity: ${_formatDefectType((defectSummary['severity'] as String?) ?? '')}',
                        style: bodyStyle,
                      ),
                  ],
                ),
              );

              if (defectTypes.isNotEmpty) {
                widgets.add(pw.SizedBox(height: 8));
                widgets.add(
                  pw.Table(
                    border: pw.TableBorder.symmetric(
                      inside: const pw.BorderSide(
                        color: PdfColors.grey400,
                        width: 0.3,
                      ),
                      outside: const pw.BorderSide(
                        color: PdfColors.grey400,
                        width: 0.5,
                      ),
                    ),
                    columnWidths: const {
                      0: pw.FlexColumnWidth(3),
                      1: pw.FlexColumnWidth(1),
                    },
                    children: [
                      pw.TableRow(
                        decoration: const pw.BoxDecoration(
                          color: PdfColors.blueGrey700,
                        ),
                        children: [
                          pw.Padding(
                            padding: const pw.EdgeInsets.symmetric(
                              vertical: 4,
                              horizontal: 6,
                            ),
                            child: pw.Text(
                              'Defect type',
                              style: tableHeaderStyle,
                            ),
                          ),
                          pw.Padding(
                            padding: const pw.EdgeInsets.symmetric(
                              vertical: 4,
                              horizontal: 6,
                            ),
                            child: pw.Text('Count', style: tableHeaderStyle),
                          ),
                        ],
                      ),
                      ...defectTypes.entries.map(
                        (entry) => pw.TableRow(
                          children: [
                            pw.Padding(
                              padding: const pw.EdgeInsets.symmetric(
                                vertical: 4,
                                horizontal: 6,
                              ),
                              child:
                                  pw.Text(entry.key, style: bodyStyle),
                            ),
                            pw.Padding(
                              padding: const pw.EdgeInsets.symmetric(
                                vertical: 4,
                                horizontal: 6,
                              ),
                              child: pw.Text(
                                entry.value.toString(),
                                style: bodyStyle,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              }
            }

            if (detections.isNotEmpty) {
              widgets.add(pw.SizedBox(height: 18));
              widgets.add(pw.Text('Detections', style: sectionTitleStyle));
              widgets.add(pw.SizedBox(height: 6));
              widgets.add(
                pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: detections.asMap().entries.map((entry) {
                    final index = entry.key + 1;
                    final detection =
                        Map<String, dynamic>.from(entry.value as Map);
                    final type = _formatDefectType(
                      (detection['defect_type'] as String?) ?? 'Unknown',
                    );
                    final confidence =
                        ((detection['confidence'] as num?)?.toDouble() ?? 0.0) *
                            100;
                    return pw.Bullet(
                      text:
                          '$index. $type — ${confidence.toStringAsFixed(1)}% confidence',
                      style: bodyStyle,
                    );
                  }).toList(),
                ),
              );
            }

            return widgets;
          },
        ),
      );

      final pdfBytes = await doc.save();

      String sanitizedPrediction = prediction.prediction
          .toLowerCase()
          .replaceAll(RegExp(r'[^a-z0-9]+'), '_');
      sanitizedPrediction =
          sanitizedPrediction.replaceAll(RegExp('_+'), '_').replaceAll(RegExp(r'^_|_$'), '');
      final timestamp =
          '${now.year}${now.month.toString().padLeft(2, '0')}${now.day.toString().padLeft(2, '0')}_${now.hour.toString().padLeft(2, '0')}${now.minute.toString().padLeft(2, '0')}';
      final baseFileName = [
        'beanscan_result',
        if (sanitizedPrediction.isNotEmpty) sanitizedPrediction,
        timestamp,
      ].join('_');
      final fileName = '$baseFileName.pdf';

      final tempDir = await getTemporaryDirectory();
      final tempPath = '${tempDir.path}${Platform.pathSeparator}$fileName';
      final tempFile = File(tempPath);
      await tempFile.writeAsBytes(pdfBytes, flush: true);

      bool savedViaDialog = false;

      if (!silent && (Platform.isAndroid || Platform.isIOS)) {
        try {
          final params = SaveFileDialogParams(
            sourceFilePath: tempFile.path,
            fileName: fileName,
            mimeTypesFilter: const ['application/pdf'],
          );
          final savedPath = await FlutterFileDialog.saveFile(params: params);
          if (savedPath != null && savedPath.isNotEmpty) {
            messenger?.showSnackBar(
              SnackBar(
                content: Text('Report saved: $savedPath'),
                duration: const Duration(seconds: 5),
              ),
            );
            savedViaDialog = true;
          }
        } catch (e, stackTrace) {
          _logResultsPage(
            'Save dialog failed',
            error: e,
            stackTrace: stackTrace,
          );
        }
      }

      if (savedViaDialog) {
        try {
          if (await tempFile.exists()) {
            await tempFile.delete();
          }
        } catch (e, stackTrace) {
          _logResultsPage(
            'Failed to delete temp PDF after dialog save',
            error: e,
            stackTrace: stackTrace,
          );
        }
        return;
      }

      File? savedFile;
      bool savedToPublicDownloads = false;
      bool usedFallbackStorage = false;

      Directory? primaryDirectory;

      if (Platform.isAndroid) {
        if (!await _ensureAndroidStoragePermissions(messenger)) {
          return;
        }

        final downloadsDir = await _getAndroidDownloadsDirectory();
        if (downloadsDir != null) {
          primaryDirectory = Directory(
            '${downloadsDir.path}${Platform.pathSeparator}BeanScan Reports',
          );
          savedToPublicDownloads = true;
        } else {
          usedFallbackStorage = true;
          _logResultsPage('Downloads directory not accessible; will use app documents directory.');
        }
      }

      final fallbackDirectory = await getApplicationDocumentsDirectory();
      primaryDirectory ??= fallbackDirectory;

      savedFile = await _tryWritePdf(pdfBytes, primaryDirectory, fileName);

      if (savedFile == null && savedToPublicDownloads) {
        usedFallbackStorage = true;
        savedToPublicDownloads = false;
        savedFile = await _tryWritePdf(pdfBytes, fallbackDirectory, fileName);
      }

      savedFile ??= await _tryWritePdf(pdfBytes, fallbackDirectory, fileName);

      if (savedFile == null) {
        throw Exception('Unable to save the PDF report.');
      }

      final message = savedToPublicDownloads
          ? 'Report saved to ${savedFile.path}'
          : usedFallbackStorage
              ? 'Downloads unavailable; report saved to ${savedFile.path}'
              : 'Report saved to ${savedFile.path}';

      messenger?.showSnackBar(
        SnackBar(
          content: Text(message),
          duration: const Duration(seconds: 5),
        ),
      );

      if (!silent) {
        try {
          await Share.shareXFiles(
            [XFile(savedFile.path)],
            text: 'BeanScan report for ${prediction.prediction}',
            subject: 'BeanScan Analysis Report',
          );
        } catch (e, stackTrace) {
          _logResultsPage(
            'Failed to share PDF report',
            error: e,
            stackTrace: stackTrace,
          );
        }
      }

      try {
        if (await tempFile.exists()) {
          await tempFile.delete();
        }
      } catch (e, stackTrace) {
        _logResultsPage(
          'Failed to delete temp PDF',
          error: e,
          stackTrace: stackTrace,
        );
      }
    } catch (e, stackTrace) {
      _logResultsPage(
        'Failed to generate PDF',
        error: e,
        stackTrace: stackTrace,
      );
      messenger?.showSnackBar(
        SnackBar(
          content: Text('Could not save PDF report: $e'),
          duration: const Duration(seconds: 4),
        ),
      );
    }
  }

  Future<bool> _ensureAndroidStoragePermissions(ScaffoldMessengerState? messenger) async {
    try {
      var storageStatus = await Permission.storage.status;
      if (!storageStatus.isGranted) {
        storageStatus = await Permission.storage.request();
      }

      PermissionStatus? manageStatus;
      if (!storageStatus.isGranted) {
        try {
          manageStatus = await Permission.manageExternalStorage.status;
          if (!manageStatus.isGranted) {
            manageStatus = await Permission.manageExternalStorage.request();
          }
        } catch (e, stackTrace) {
          _logResultsPage(
            'manageExternalStorage permission check failed',
            error: e,
            stackTrace: stackTrace,
          );
        }
      }

      final hasAccess =
          storageStatus.isGranted || (manageStatus?.isGranted ?? false);

      if (!hasAccess) {
        final permanentlyDenied =
            storageStatus.isPermanentlyDenied || (manageStatus?.isPermanentlyDenied ?? false);
        messenger?.showSnackBar(
          SnackBar(
            content: const Text('Storage permission is required to save reports to Downloads.'),
            action: permanentlyDenied
                ? SnackBarAction(
                    label: 'Settings',
                    onPressed: () => openAppSettings(),
                  )
                : null,
            duration: const Duration(seconds: 4),
          ),
        );
      }

      return hasAccess;
    } catch (e, stackTrace) {
      _logResultsPage(
        'Failed to request storage permissions',
        error: e,
        stackTrace: stackTrace,
      );
      messenger?.showSnackBar(
        SnackBar(
          content: Text('Unable to request storage permission: $e'),
          duration: const Duration(seconds: 4),
        ),
      );
      return false;
    }
  }

  Future<Directory?> _getAndroidDownloadsDirectory() async {
    final List<String> candidatePaths = [];

    void addCandidate(String path) {
      if (path.isEmpty) return;
      if (!candidatePaths.contains(path)) {
        candidatePaths.add(path);
      }
    }

    const manualFallbacks = [
      '/storage/emulated/0/Download',
      '/storage/emulated/0/Downloads',
      '/sdcard/Download',
      '/sdcard/Downloads',
    ];
    for (final path in manualFallbacks) {
      addCandidate(path);
    }

    try {
      final externalDirs = await getExternalStorageDirectories();
      if (externalDirs != null) {
        for (final dir in externalDirs) {
          final path = dir.path;
          final androidIndex = path.indexOf('${Platform.pathSeparator}Android${Platform.pathSeparator}');
          if (androidIndex != -1) {
            final root = path.substring(0, androidIndex);
            addCandidate('$root${Platform.pathSeparator}Download');
            addCandidate('$root${Platform.pathSeparator}Downloads');
          }
        }
      }
    } catch (e, stackTrace) {
      _logResultsPage(
        'Error sampling external storage root for downloads',
        error: e,
        stackTrace: stackTrace,
      );
    }

    try {
      final downloadsDirs = await getExternalStorageDirectories(
        type: StorageDirectory.downloads,
      );
      if (downloadsDirs != null) {
        for (final dir in downloadsDirs) {
          addCandidate(dir.path);
        }
      }
    } catch (e, stackTrace) {
      _logResultsPage(
        'Error fetching external downloads directories',
        error: e,
        stackTrace: stackTrace,
      );
    }

    for (final path in candidatePaths) {
      final directory = Directory(path);
      try {
        if (await directory.exists()) {
          return directory;
        }
      } catch (e, stackTrace) {
        _logResultsPage(
          'Error checking downloads directory candidate "$path"',
          error: e,
          stackTrace: stackTrace,
        );
      }
    }

    return null;
  }

  Future<File?> _tryWritePdf(Uint8List bytes, Directory directory, String fileName) async {
    try {
      await directory.create(recursive: true);
      final filePath = '${directory.path}${Platform.pathSeparator}$fileName';
      final file = File(filePath);
      await file.writeAsBytes(bytes, flush: true);
      return file;
    } catch (e, stackTrace) {
      _logResultsPage(
        'Failed to write PDF to ${directory.path}',
        error: e,
        stackTrace: stackTrace,
      );
      return null;
    }
  }

  Widget _buildImagePreview(BuildContext context, ColorScheme colorScheme) {
    if (imagePath.isEmpty) {
      return Container(
        height: 220,
        width: double.infinity,
        decoration: BoxDecoration(
          color: colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(AppConstants.largeRadius),
          border: Border.all(
            color: colorScheme.outline.withValues(alpha: 0.3),
            width: AppConstants.thinBorder,
          ),
        ),
        clipBehavior: Clip.antiAlias,
        child: Center(
          child: Icon(Icons.image, color: colorScheme.onSurface.withValues(alpha: 0.54), size: 48),
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
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(
          color: colorScheme.outline.withValues(alpha: 0.3),
          width: AppConstants.thinBorder,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        child: imageStack,
      ),
    );
  }

  Widget _buildImageWidget(BuildContext context, ColorScheme colorScheme, {BoxFit fit = BoxFit.cover}) {
    _logResultsPage('🔍 _buildImageWidget - imagePath: $imagePath');
    final isHttp = imagePath.startsWith('http');
    final isAbsolutePath = imagePath.startsWith('/') || imagePath.startsWith('http');
    final String url = imagePath.startsWith('/') ? (ApiService.apiUrl + imagePath) : imagePath;
    _logResultsPage('🔍 _buildImageWidget - isHttp: $isHttp, isAbsolutePath: $isAbsolutePath, url: $url');
    
    if (isHttp || imagePath.startsWith('/')) {
      return Image.network(
        url,
        fit: fit,
        width: double.infinity,
        height: double.infinity,
        errorBuilder: (c, e, s) {
          _logResultsPage('🔍 Image.network error: $e');
          return Center(
            child: Icon(
              Icons.broken_image,
              color: colorScheme.onSurface.withValues(alpha: 0.54),
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
        _logResultsPage('🔍 Image.file error: $e');
        return Center(
          child: Icon(
            Icons.broken_image,
            color: colorScheme.onSurface.withValues(alpha: 0.54),
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
    _logResultsPage('Defect detections: ${detections.length}');
    for (int i = 0; i < detections.length; i++) {
      _logResultsPage('Detection $i: ${detections[i]}');
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
          color: Colors.red.withValues(alpha: 0.8),
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

  Widget _buildInfoCard(BuildContext context, ColorScheme colorScheme, TextTheme textTheme) {
    final DateTime now = DateTime.now();
    final String dateStr = '${now.month}/${now.day}/${now.year} - ${now.hour}:${now.minute.toString().padLeft(2, '0')}';
    final double healthyPct = (prediction.confidence * 100).clamp(0.0, 100.0);
    final Map<String, dynamic>? shelfLifeData = shelfLife != null
        ? Map<String, dynamic>.from(shelfLife!)
        : null;
    double? predictedDays = _asDouble(shelfLifeData?['predicted_days']);
    double? estimatedMonths = _asDouble(shelfLifeData?['estimated_months']);
    if (estimatedMonths == null && predictedDays != null && predictedDays > 0) {
      estimatedMonths = double.parse((predictedDays / 30.0).toStringAsFixed(1));
    }
    final Map<String, dynamic>? monthsRange = _normalizeMonthsRange(
      shelfLifeData?['estimated_months_range'] is Map
          ? Map<String, dynamic>.from(shelfLifeData!['estimated_months_range'] as Map)
          : null,
      estimatedMonths,
    );
    final int predictedDaysDisplay =
        (predictedDays ?? _asDouble(shelfLifeData?['predicted_days']) ?? 0).round();
    final Map<String, dynamic>? defectSummary = _getDefectSummary();
    final String? severityLabel = (shelfLifeData?['severity'] as String?) ?? (defectSummary?['severity'] as String?);
    final double confidenceScore = shelfLifeData != null
        ? (_asDouble(shelfLifeData['confidence_score'] ?? shelfLifeData['confidence']) ?? 0.0)
        : (healthyPct / 100.0);
    final String statusLabel = _resolveStatus(
      category: shelfLifeData?['category'] as String?,
      severity: severityLabel,
      predictedDays: predictedDays,
    );
    final onSurface = colorScheme.onSurface;
    final surface = colorScheme.surface;
    final isDark = colorScheme.brightness == Brightness.dark;
    final borderColor = isDark ? colorScheme.outline.withValues(alpha: 0.4) : AppColors.dividerGrey;
    final primaryTextColor = isDark ? colorScheme.onSurface : AppColors.textDarkGrey;
    final dividerColor = isDark ? colorScheme.outline.withValues(alpha: 0.3) : AppColors.dividerGrey;
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
        border: Border.all(color: borderColor, width: AppConstants.thinBorder),
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
                  style: TextStyle(color: onSurface.withValues(alpha: 0.7)),
                ),
              ),
              IconButton(
                onPressed: () => _downloadResultPdf(context),
                icon: Icon(Icons.download, size: 18, color: onSurface.withValues(alpha: 0.7)),
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
                style: TextStyle(fontWeight: FontWeight.w600, color: onSurface.withValues(alpha: 0.75)),
              ),
              Text(
                prediction.prediction,
                style: TextStyle(color: onSurface),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Divider(
            color: dividerColor,
            height: 24,
          ),
          Text(
            'Estimated Shelf Life',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: onSurface.withValues(alpha: 0.8),
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
                  style: TextStyle(fontWeight: FontWeight.w500, color: onSurface.withValues(alpha: 0.75)),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: chipBackground,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Text(
                    '$predictedDaysDisplay days',
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
                Text(
                  'Estimated Months:',
                  style: TextStyle(
                    fontWeight: FontWeight.w500,
                    color: primaryTextColor.withValues(alpha: 0.75),
                  ),
                ),
                Text(
                  _formatMonthsText(estimatedMonths, monthsRange),
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: primaryTextColor,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
          ],
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Status:',
                style: TextStyle(
                  fontWeight: FontWeight.w500,
                  color: primaryTextColor.withValues(alpha: 0.75),
                ),
              ),
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
              Text(
                'Confidence Score:',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: primaryTextColor,
                ),
              ),
              Text(
                '${(confidenceScore * 100).clamp(0, 100).toStringAsFixed(0)}%',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: primaryTextColor,
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
    final isDark = colorScheme.brightness == Brightness.dark;
    final borderColor = isDark ? colorScheme.outline.withValues(alpha: 0.4) : AppColors.dividerGrey;
    final primaryTextColor = isDark ? colorScheme.onSurface : AppColors.textDarkGrey;
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: borderColor, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.bug_report, color: colorScheme.primary, size: 20),
              const SizedBox(width: 8),
              Text(
                'Defect Detection Results',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: primaryTextColor,
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
              Text(
                'Total Defects:',
                style: TextStyle(fontWeight: FontWeight.w500, color: primaryTextColor.withValues(alpha: 0.8)),
              ),
              Text(
                '${summary['total_defects'] ?? 0}',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: primaryTextColor,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // Defect Types
          if (summary['defect_types'] != null && (summary['defect_types'] as Map).isNotEmpty) ...[
            Text(
              'Defect Types:',
              style: TextStyle(
                fontWeight: FontWeight.w500,
                color: primaryTextColor.withValues(alpha: 0.8),
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 4,
              children: (Map<String, dynamic>.from(summary['defect_types'] as Map)).entries
                  .map((entry) => Chip(
                        label: Text('${entry.key}: ${entry.value}'),
                        backgroundColor: colorScheme.secondary.withValues(alpha: isDark ? 0.35 : 0.2),
                        labelStyle: TextStyle(
                          fontSize: 12,
                          color: isDark ? colorScheme.onSecondaryContainer : colorScheme.onSecondary,
                        ),
                      ))
                  .toList(),
            ),
            const SizedBox(height: 12),
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

  double? _asDouble(dynamic value) {
    if (value is num) return value.toDouble();
    if (value is String) {
      final trimmed = value.trim();
      if (trimmed.isEmpty) return null;
      final cleaned = trimmed.replaceAll(RegExp(r'[^0-9\.\-]'), '');
      if (cleaned.isEmpty) return null;
      return double.tryParse(cleaned);
    }
    return null;
  }

  int? _asInt(dynamic value) {
    final double? parsed = _asDouble(value);
    return parsed?.round();
  }

  Map<String, dynamic>? _normalizeMonthsRange(Map<String, dynamic>? range, double? _) {
    if (range != null) {
      final double? min = _asDouble(range['min']);
      final double? max = _asDouble(range['max']);
      if (min != null && max != null && min > 0 && max > 0) {
        return {'min': min, 'max': max};
      }
    }
    return null;
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
    String? normalizedSeverity = _normalizeSeverityTag(severity);
    final String? rawCategory = category?.trim().isNotEmpty == true ? category!.trim() : null;
    final String? normalizedCategory = _normalizeSeverityTag(rawCategory);

    if ((normalizedSeverity == null || normalizedSeverity.isEmpty) && predictedDays != null) {
      final double predicted = predictedDays.toDouble();
      final double normalizedPct = predicted > 0 ? (predicted / 240.0) * 100.0 : 0;
      normalizedSeverity = _computeSeverityFromPercentage(normalizedPct);
    }

    final int severityRank = _severityRank(normalizedSeverity);
    final int categoryRank = _severityRank(normalizedCategory);

    if (severityRank == 0 && categoryRank == 0) {
      if (rawCategory != null && rawCategory.isNotEmpty) {
        return _capitalize(rawCategory.toLowerCase());
      }
      if (normalizedSeverity != null && normalizedSeverity.isNotEmpty) {
        return _statusLabelFromSeverity(normalizedSeverity);
      }
      return 'Unknown';
    }

    if (categoryRank > severityRank && rawCategory != null && rawCategory.isNotEmpty) {
      return _capitalize(rawCategory.toLowerCase());
    }

    if (categoryRank == severityRank && categoryRank > 0 && rawCategory != null && rawCategory.isNotEmpty) {
      return _capitalize(rawCategory.toLowerCase());
    }

    if (normalizedSeverity != null && normalizedSeverity.isNotEmpty) {
      return _statusLabelFromSeverity(normalizedSeverity);
    }

    if (rawCategory != null && rawCategory.isNotEmpty) {
      return _capitalize(rawCategory.toLowerCase());
    }

    return 'Unknown';
  }

  String? _normalizeSeverityTag(String? value) {
    final String? trimmed = value?.trim();
    if (trimmed == null || trimmed.isEmpty) {
      return null;
    }

    final String lower = trimmed.toLowerCase();
    switch (lower) {
      case 'mild':
      case 'moderate':
      case 'severe':
        return lower;
      case 'excellent':
      case 'good':
      case 'optimal':
      case 'great':
      case 'low':
      case 'best':
        return 'mild';
      case 'warning':
      case 'fair':
      case 'medium':
      case 'elevated':
      case 'moderate risk':
        return 'moderate';
      case 'critical':
      case 'expired':
      case 'poor':
      case 'high':
      case 'extreme':
      case 'severe risk':
        return 'severe';
      default:
        return null;
    }
  }

  int _severityRank(String? normalizedSeverity) {
    switch (normalizedSeverity) {
      case 'mild':
        return 1;
      case 'moderate':
        return 2;
      case 'severe':
        return 3;
      default:
        return 0;
    }
  }

  String _statusLabelFromSeverity(String normalizedSeverity) {
    switch (normalizedSeverity) {
      case 'mild':
        return 'Excellent';
      case 'moderate':
        return 'Warning';
      case 'severe':
        return 'Critical';
      default:
        return _capitalize(normalizedSeverity);
    }
  }

  String _capitalize(String value) {
    if (value.isEmpty) return value;
    return value[0].toUpperCase() + value.substring(1).toLowerCase();
  }


  
  Widget _buildSeverityAndDefectiveTiles(ColorScheme colorScheme) {
    final summary = _getDefectSummary();
    final List<dynamic> detections = _getDetections();
    final double? shelfLifePct =
        _asDouble(shelfLife?['defect_percentage'] ?? shelfLife?['defective_percent']);
    final int totalDefects =
        summary?['total_defects'] is num ? (summary!['total_defects'] as num).toInt() : detections.length;
    final double? summaryPct = _asDouble(summary?['defect_percentage']);

    double? resolvedPctCandidate = shelfLifePct;
    if (resolvedPctCandidate == null || (resolvedPctCandidate <= 0 && summaryPct != null && summaryPct > 0)) {
      resolvedPctCandidate = summaryPct;
    }

    double resolvedPctValue;
    if (totalDefects > 0) {
      double? detectionDrivenPct = resolvedPctCandidate;
      if (detectionDrivenPct == null || detectionDrivenPct <= 0) {
        detectionDrivenPct = _estimateDefectPercentageFromDetections(detections) ??
            (totalDefects / math.max(totalDefects, 12)) * 100.0;
      }
      resolvedPctValue = math.max(detectionDrivenPct, 28.0);
    } else {
      if (resolvedPctCandidate != null) {
        resolvedPctValue = resolvedPctCandidate;
      } else {
        final double lowConfidencePenalty = ((1.0 - prediction.confidence).clamp(0.0, 1.0)) * 40.0;
        resolvedPctValue = lowConfidencePenalty;
      }
    }

    final double defectivePct = resolvedPctValue.clamp(0.0, 100.0);

    final String computedSeverity = _computeSeverityFromPercentage(defectivePct);
    final String? rawSeverity = summary?['severity'] as String? ?? (shelfLife?['severity'] as String?);
    final String? normalizedSeverity = _normalizeSeverityTag(rawSeverity);
    String severityLabel;
    if (normalizedSeverity == null || normalizedSeverity.isEmpty) {
      severityLabel = computedSeverity;
    } else {
      final int existingRank = _severityRank(normalizedSeverity);
      final int computedRank = _severityRank(computedSeverity);
      severityLabel = computedRank > existingRank ? computedSeverity : normalizedSeverity;
    }

    int severityLevel;
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
        severityLevel = defectivePct < 25 ? 1 : (defectivePct < 45 ? 2 : 3);
    }

    if (totalDefects > 0 && severityLevel == 1) {
      severityLevel = 2;
      severityLabel = 'moderate';
    }

    return Row(
      children: [
        Expanded(
          child: _severityCard(
            colorScheme,
            severityLevel,
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
    if (defectDetection == null && shelfLife == null) return null;

    Map<String, dynamic> summary = {};
    if (defectDetection?['summary'] is Map<String, dynamic>) {
      summary = Map<String, dynamic>.from(defectDetection!['summary'] as Map);
    } else if (defectDetection != null) {
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

    final Map<String, dynamic>? shelfLifeData =
        shelfLife != null ? Map<String, dynamic>.from(shelfLife!) : null;
    final double? shelfLifePct =
        _asDouble(shelfLifeData?['defect_percentage'] ?? shelfLifeData?['defective_percent']);

    final detections = _getDetections();
    if (!summary.containsKey('total_defects') ||
        summary['total_defects'] == null ||
        (_asInt(summary['total_defects']) ?? 0) == 0 && detections.isNotEmpty) {
      summary['total_defects'] = detections.length;
    }

    final Map<String, int> typeCounts = {};
    for (final detection in detections) {
      final rawType = (detection['defect_type'] as String? ?? 'Unknown').toLowerCase();
      typeCounts[rawType] = (typeCounts[rawType] ?? 0) + 1;
    }

    final dynamic shelfLifeCountsRaw = shelfLifeData?['defect_counts'];
    if (shelfLifeCountsRaw is Map) {
      final shelfLifeCounts = Map<String, dynamic>.from(shelfLifeCountsRaw);
      for (final entry in shelfLifeCounts.entries) {
        final key = entry.key.toString().toLowerCase();
        final value = entry.value;
        if (value is num) {
          final current = typeCounts[key] ?? 0;
          typeCounts[key] = math.max(current, value.toInt());
        }
      }
    }
    if (typeCounts.isNotEmpty) {
      summary['defect_types'] =
          typeCounts.map((key, value) => MapEntry(_formatDefectType(key), value));
      final int countsTotal = typeCounts.values.fold(0, (prev, value) => prev + value);
      summary['total_defects'] = math.max(_asInt(summary['total_defects']) ?? 0, countsTotal);
    }

    final String? currentQuality = summary['quality_grade'] as String?;
    if (!summary.containsKey('quality_grade') || currentQuality == null || currentQuality.isEmpty) {
      final quality = shelfLifeData?['quality_grade'];
      if (quality is String && quality.isNotEmpty) {
        summary['quality_grade'] = quality;
      }
    }

    double? summaryPct = _asDouble(summary['defect_percentage']);
    if (summaryPct == null || (summaryPct == 0 && shelfLifePct != null && shelfLifePct > 0)) {
      summaryPct = shelfLifePct;
    }

    String? severity = summary['severity'] as String?;
    if (severity == null || severity.isEmpty) {
      final dynamic shelfSeverity = shelfLifeData?['severity'] ?? shelfLifeData?['category'];
      if (shelfSeverity is String && shelfSeverity.isNotEmpty) {
        severity = shelfSeverity;
      }
    }

    double summaryPctValue = summaryPct ?? 0;
    if (summaryPct == null) {
      final double pctForSeverity = shelfLifePct ??
          (_asInt(summary['total_defects']) ?? detections.length) * 8.0;
      final String severityValue = severity ??= _computeSeverityFromPercentage(pctForSeverity);
      double fallbackPercentage;
      switch (severityValue.toLowerCase()) {
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
      summaryPctValue = fallbackPercentage;
    }

    final double cappedPct = summaryPctValue.clamp(0.0, 100.0);
    summary['defect_percentage'] = cappedPct;
    final String normalizedSeverity =
        _normalizeSeverityTag(severity) ?? _computeSeverityFromPercentage(cappedPct);
    summary['severity'] = normalizedSeverity;

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

  double? _estimateDefectPercentageFromDetections(List<dynamic> detections) {
    if (detections.isEmpty) {
      return null;
    }

    double totalConfidence = 0;
    for (final detection in detections) {
      final double confidence = (detection['confidence'] as num?)?.toDouble() ?? 0.5;
      totalConfidence += confidence.clamp(0.0, 1.0);
    }

    final double averageConfidence = totalConfidence / detections.length;
    final double normalizedConfidence = averageConfidence.clamp(0.3, 0.95);
    final double countFactor = math.min(1.0, detections.length / 3.0);
    final double base = 32.0 + (detections.length * 18.0).clamp(0.0, 54.0);
    final double confidenceAdjustment = (normalizedConfidence - 0.5) * 50.0;
    final double countAdjustment = countFactor * 35.0;
    final double estimated = base + confidenceAdjustment + countAdjustment;
    return estimated.clamp(24.0, 95.0);
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

  Widget _severityCard(ColorScheme colorScheme, int severityLevel, {String? severityLabel}) {
    final computedLabel = severityLabel != null && severityLabel.isNotEmpty
        ? _formatSeverityLabel(severityLabel)
        : (severityLevel == 1 ? 'Mild' : (severityLevel == 2 ? 'Moderate' : 'Severe'));
    final isDark = colorScheme.brightness == Brightness.dark;
    final borderColor = isDark ? colorScheme.outline.withValues(alpha: 0.4) : AppColors.dividerGrey;
    final primaryTextColor = isDark ? colorScheme.onSurface : AppColors.textDarkGrey;
    return Container(
      padding: const EdgeInsets.all(AppConstants.largePadding),
      constraints: const BoxConstraints(minHeight: 220),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: borderColor, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Severity:',
            style: TextStyle(fontWeight: FontWeight.w600, color: primaryTextColor),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 140,
            width: double.infinity,
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  BeanSeverityIcon(severityLevel: severityLevel, size: 72, color: colorScheme.primary),
                  const SizedBox(height: AppConstants.smallSpacing),
                  Text(
                    computedLabel,
                    style: TextStyle(color: primaryTextColor, fontWeight: FontWeight.w600),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _defectivePercentCard(ColorScheme colorScheme, double percent) {
    percent = percent.clamp(0, 100);
    final isDark = colorScheme.brightness == Brightness.dark;
    final borderColor = isDark ? colorScheme.outline.withValues(alpha: 0.4) : AppColors.dividerGrey;
    final primaryTextColor = isDark ? colorScheme.onSurface : AppColors.textDarkGrey;
    return Container(
      padding: const EdgeInsets.all(AppConstants.largePadding),
      constraints: const BoxConstraints(minHeight: 220),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: borderColor, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
           Text(
             'Defective (%)',
             style: TextStyle(fontWeight: FontWeight.w600, color: primaryTextColor),
           ),
           const SizedBox(height: 16), // Increased spacing
          Center(child: _circularPercent(colorScheme, percent: percent, color: colorScheme.primary)),
          const SizedBox(height: 16), // Added bottom spacing
        ],
      ),
    );
  }

  Widget _circularPercent(ColorScheme colorScheme, {required double percent, required Color color}) {
    _logResultsPage('Circular percent widget - percent: $percent, color: $color');
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
               backgroundColor: colorScheme.surfaceContainerHighest,
               valueColor: AlwaysStoppedAnimation<Color>(color),
             ),
           ),
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: colorScheme.surface,
              borderRadius: BorderRadius.circular(36),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black12,
                  blurRadius: 4,
                ),
              ],
            ),
            alignment: Alignment.center,
            child: Text(
              '${percent.toStringAsFixed(0)}%',
              style: TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 18,
                color: colorScheme.onSurface,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildYesNoButtons(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).pop(ResultsNavigationAction.scan),
            style: ElevatedButton.styleFrom(
              backgroundColor: colorScheme.primary,
              foregroundColor: colorScheme.onPrimary,
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
              backgroundColor: colorScheme.surfaceContainerHighest,
              foregroundColor: colorScheme.onSurface,
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
    _logResultsPage('Painting defects on canvas size: $size');
    _logResultsPage('Total detections received: ${detections.length}');
    
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
    
    _logResultsPage('Detected source size: $sourceSize -> scaleX=$scaleX, scaleY=$scaleY');

    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];
      final coordinates = detection['coordinates'] as Map<String, dynamic>?;
      if (coordinates == null) {
        _logResultsPage('Detection $i: No coordinates found');
        continue;
      }

      final x1 = coordinates['x1'] as double? ?? 0.0;
      final y1 = coordinates['y1'] as double? ?? 0.0;
      final x2 = coordinates['x2'] as double? ?? 0.0;
      final y2 = coordinates['y2'] as double? ?? 0.0;
      final defectType = detection['defect_type'] as String? ?? 'Unknown';
      final confidence = detection['confidence'] as double? ?? 0.0;

      _logResultsPage('Detection $i: $defectType at ($x1, $y1, $x2, $y2) with confidence $confidence');

      // Check if coordinates are valid (not all zeros)
      if ((x1 == 0.0 && y1 == 0.0 && x2 == 0.0 && y2 == 0.0) || x2 <= x1 || y2 <= y1) {
        _logResultsPage('Detection $i: Invalid coordinates, skipping visual overlay but keeping in counts.');
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
        _logResultsPage('Detection $i: Bounding box very small after scaling ($boxWidth x $boxHeight), drawing indicator dot instead');
        final dotPaint = Paint()
          ..color = paint.color
          ..style = PaintingStyle.fill;
        final dotCenter = Offset(finalX1.clamp(4.0, size.width - 4.0), finalY1.clamp(4.0, size.height - 4.0));
        canvas.drawCircle(dotCenter, 4, dotPaint);

        // Removed numeric badge near tiny detections to avoid duplication

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
          ..color = Colors.black.withValues(alpha: 0.7)
          ..style = PaintingStyle.fill;
        canvas.drawRect(labelRect, labelPaint);
        textPainter.paint(canvas, Offset(labelRect.left + 4, labelRect.top + 2));

        validBoxesDrawn++;
        continue;
      }
      
      _logResultsPage('Detection $i: Scaled from ($x1, $y1, $x2, $y2) to ($finalX1, $finalY1, $finalX2, $finalY2)');

      // Draw bounding box (outline only) using scaled coordinates
      final rect = Rect.fromLTRB(finalX1, finalY1, finalX2, finalY2);
      canvas.drawRect(rect, paint);
      validBoxesDrawn++;

      // Removed numeric badge on boxes to avoid double counting visuals

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
        ..color = Colors.black.withValues(alpha: 0.6)
        ..style = PaintingStyle.fill;

      canvas.drawRect(labelRect, labelPaint);
      textPainter.paint(canvas, Offset(finalX1 + 4, labelRect.top + 2));
    }
    
    _logResultsPage('Valid bounding boxes drawn: $validBoxesDrawn out of ${detections.length} total detections');
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


