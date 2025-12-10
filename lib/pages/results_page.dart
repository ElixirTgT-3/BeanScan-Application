import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

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
  static final Map<String, Future<Size?>> _imageSizeCache = <String, Future<Size?>>{};
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

  const ResultsPage.history({
    Key? key,
    required BeanPrediction prediction,
    required String imagePath,
    Map<String, dynamic>? defectDetection,
    Map<String, dynamic>? shelfLife,
  }) : this(
          key: key,
          prediction: prediction,
          imagePath: imagePath,
          defectDetection: defectDetection,
          shelfLife: shelfLife,
          shouldAutoSave: false,
          fromHistory: true,
        );

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
                _buildActionButtons(context, colorScheme),
                const SizedBox(height: AppConstants.largeSpacing),
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
      Map<String, dynamic>? shelfLifeData =
          shelfLife != null ? Map<String, dynamic>.from(shelfLife!) : null;
      final defectSummary = _getDefectSummary();
      final detections = _getDetections();
      final bool hasGoodBeansOnly =
          _hasGoodBeans(detections) && ((defectSummary?['total_defects'] as num? ?? 0) == 0);
      if (hasGoodBeansOnly) {
        const double fallbackDays = 1095.0;
        final double fallbackMonths = double.parse((fallbackDays / 30.0).toStringAsFixed(1));
        shelfLifeData ??= {};
        final double? currentDays = _asDouble(shelfLifeData['predicted_days']);
        final double? currentMonths = _asDouble(shelfLifeData['estimated_months']);
        if (currentDays == null || currentDays <= 0) {
          shelfLifeData['predicted_days'] = fallbackDays;
        }
        if (currentMonths == null || currentMonths <= 0) {
          shelfLifeData['estimated_months'] = fallbackMonths;
        }
        shelfLifeData.putIfAbsent('base_shelf_life', () => fallbackDays);
        shelfLifeData.putIfAbsent('category', () => 'Excellent');
        shelfLifeData.putIfAbsent('severity', () => 'normal');
      }
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
              final double? predictedDaysRaw = _asDouble(shelfLifeData['predicted_days']);
              final double? weightedDays = _weightedShelfLifeDays(
                typeCounts: defectSummary?['defect_types'] as Map<String, dynamic>?,
                detections: detections,
              );
              double predictedDays = weightedDays ??
                  (predictedDaysRaw != null && predictedDaysRaw > 0
                      ? predictedDaysRaw
                      : 30.0);
              double? estimatedMonths = double.parse((predictedDays / 30.0).toStringAsFixed(1));
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
              final String severityLabelPdf =
                  (shelfLifeData['severity'] as String?) ?? (defectSummary?['severity'] as String?) ?? 'normal';
              final String status = _resolveStatus(
                category: (shelfLifeData['category'] as String?) ??
                    _deriveShelfLifeCategory({'predicted_days': predictedDays}),
                severity: severityLabelPdf,
                predictedDays: predictedDays,
                baseDays: weightedDays ?? predictedDaysRaw ?? 30.0,
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

    final List<dynamic> detections = _getDetections();
        // Get backend-provided image dimensions first (coordinates are in this space)
    final Size? detectionSize = _resolveOriginalDetectionSize(detections);

        // For camera images, EXIF orientation might cause mismatch between backend and displayed dimensions
        // Get both backend dimensions (for coordinates) and actual displayed dimensions (for rendering)
    return FutureBuilder<Size?>(
          future: _resolveImageDisplaySize(null), // Get actual displayed image size (after EXIF)
      builder: (context, snapshot) {
            final Size? actualDisplayedSize = snapshot.data; // Actual size Flutter displays (after EXIF)
        final bool hasDetections = detections.isNotEmpty;
            // CRITICAL: Backend coordinates are in backend's coordinate space (after PIL applies EXIF)
            // But Flutter might display the image at different dimensions if EXIF handling differs
            // Use backend dimensions for coordinate space, but we'll adjust if there's a mismatch
            final Size? coordinateSpaceSize = detectionSize;
            
            // Debug: Check if there's a dimension mismatch (indicates EXIF orientation issue)
            if (detectionSize != null && actualDisplayedSize != null) {
              final bool dimensionsMatch = (detectionSize.width - actualDisplayedSize.width).abs() < 1.0 &&
                                          (detectionSize.height - actualDisplayedSize.height).abs() < 1.0;
              if (!dimensionsMatch) {
                debugPrint('[ResultsPage] WARNING: Dimension mismatch detected! '
                  'Backend: ${detectionSize.width}x${detectionSize.height}, '
                  'Displayed: ${actualDisplayedSize.width}x${actualDisplayedSize.height}. '
                  'This suggests EXIF orientation mismatch between backend and Flutter.');
              }
            }
            
                // Debug: Log which dimensions we're using
            if (coordinateSpaceSize != null) {
              debugPrint('[ResultsPage] Using coordinate space: ${coordinateSpaceSize.width}x${coordinateSpaceSize.height} '
                '(detectionSize: ${detectionSize?.width}x${detectionSize?.height}, '
                'actualDisplayedSize: ${actualDisplayedSize?.width}x${actualDisplayedSize?.height})');
            }

        // Image widget will be built inside LayoutBuilder with calculated size

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
            child: LayoutBuilder(
              builder: (context, constraints) {
                // Get actual rendered container size
                final containerSize = Size(constraints.maxWidth, constraints.maxHeight);
                debugPrint('[ResultsPage] LayoutBuilder - containerSize: ${containerSize.width}x${containerSize.height}');
                
                // Calculate displayed image size based on BoxFit.contain
                // This matches what the image widget will actually render
                double displayedWidth = containerSize.width;
                double displayedHeight = containerSize.height;
                double offsetX = 0.0;
                double offsetY = 0.0;
                
                // Use actualDisplayedSize for aspect ratio if available (accounts for EXIF rotation)
                // Otherwise fall back to coordinateSpaceSize (backend dimensions)
                final Size? sizeForAspectRatio = actualDisplayedSize ?? coordinateSpaceSize;
                
                if (sizeForAspectRatio != null) {
                  final imageAspectRatio = sizeForAspectRatio.width / sizeForAspectRatio.height;
                  final containerAspectRatio = containerSize.width / containerSize.height;
                  
                  if (imageAspectRatio > containerAspectRatio) {
                    // Image is wider - fit to width
                    displayedWidth = containerSize.width;
                    displayedHeight = containerSize.width / imageAspectRatio;
                    offsetY = (containerSize.height - displayedHeight) / 2.0;
                  } else {
                    // Image is taller - fit to height
                    displayedHeight = containerSize.height;
                    displayedWidth = containerSize.height * imageAspectRatio;
                    offsetX = (containerSize.width - displayedWidth) / 2.0;
                  }
                  
                  debugPrint('[ResultsPage] Image positioning - using ${sizeForAspectRatio.width}x${sizeForAspectRatio.height} for aspect ratio, '
                    'displayed: ${displayedWidth}x${displayedHeight}, offset: ($offsetX, $offsetY)');
                }
                
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    // Image layer - positioned at calculated offset to match CustomPaint coordinates
                    // The image must be displayed at exactly the calculated size for coordinates to align
                    // Use FittedBox to ensure the image scales to fill the exact dimensions
                    Positioned(
                      left: offsetX,
                      top: offsetY,
                      child: SizedBox(
                        width: displayedWidth,
                        height: displayedHeight,
                        child: _buildImageWidget(
                          context, 
                          colorScheme, 
                          fit: BoxFit.contain,
                        ),
                      ),
                    ),
                    // Defect annotations overlay - positioned at container level to get correct size
                    // CRITICAL: The CustomPaint must receive the same container size and use the same
                    // offset calculations as the image positioning above
                    if (hasDetections)
                      IgnorePointer(
                        child: SizedBox(
                          width: containerSize.width,
                          height: containerSize.height,
                          child: CustomPaint(
                            painter: DefectAnnotationPainter(
                              _getDetections(),
                              originalSize: coordinateSpaceSize,
                              actualDisplayedSize: actualDisplayedSize,
                              imageDisplayedSize: displayedWidth > 0 && displayedHeight > 0 
                                  ? Size(displayedWidth, displayedHeight) 
                                  : null,
                              imageOffset: Offset(offsetX, offsetY),
                            ),
                          ),
                        ),
                      ),
                    // Defect count overlay
                    if (defectDetection != null && defectDetection!['summary'] != null)
                      _buildDefectCountOverlay(colorScheme),
                  ],
                );
              },
            ),
          ),
        );
      },
    );
  }

  Widget _buildImageWidget(BuildContext context, ColorScheme colorScheme, {BoxFit fit = BoxFit.cover}) {
    _logResultsPage('_buildImageWidget - imagePath: $imagePath');
    final File potentialFile = File(imagePath);
    final bool fileExists = potentialFile.existsSync();
    final bool isHttp = imagePath.startsWith('http');
    final bool isServerRelative = imagePath.startsWith('/') && !fileExists;
    final bool useNetwork = isHttp || isServerRelative;
    final String url = isServerRelative ? (ApiService.apiUrl + imagePath) : imagePath;
    _logResultsPage('_buildImageWidget - useNetwork: $useNetwork, fileExists: $fileExists, url: $url');

    if (useNetwork) {
      return Image.network(
        url,
        fit: fit,
        // Don't use infinite constraints - let parent SizedBox control size
        // width: double.infinity,
        // height: double.infinity,
        errorBuilder: (c, e, s) {
          _logResultsPage('_buildImageWidget Image.network error: $e');
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

    if (!fileExists) {
      _logResultsPage('_buildImageWidget - local file not found: $imagePath');
      return Center(
        child: Icon(
          Icons.broken_image,
          color: colorScheme.onSurface.withValues(alpha: 0.54),
          size: 48,
        ),
      );
    }

    return Image.file(
      potentialFile,
      fit: fit,
      // Don't use infinite constraints - let parent SizedBox control size
      // width: double.infinity,
      // height: double.infinity,
      errorBuilder: (c, e, s) {
        _logResultsPage('_buildImageWidget Image.file error: $e');
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

  Future<Size?> _resolveImageDisplaySize(Size? detectionSize) {
    if (detectionSize != null && detectionSize.width > 0 && detectionSize.height > 0) {
      return Future.value(detectionSize);
    }
    if (imagePath.isEmpty) {
      return Future.value(null);
    }

    final File potentialFile = File(imagePath);
    final bool fileExists = potentialFile.existsSync();
    final bool isHttp = imagePath.startsWith('http');
    final bool isServerRelative = imagePath.startsWith('/') && !fileExists;
    final bool useNetwork = isHttp || isServerRelative;
    final String resolvedSource = isServerRelative ? (ApiService.apiUrl + imagePath) : imagePath;
    final String cacheKey = useNetwork ? resolvedSource : potentialFile.path;

    final Future<Size?>? cached = _imageSizeCache[cacheKey];
    if (cached != null) {
      return cached;
    }

    final Future<Size?> loader = () async {
      try {
        final ImageProvider provider = useNetwork
            ? NetworkImage(resolvedSource)
            : FileImage(potentialFile);
        final Size size = await _decodeImageSize(provider);
        return size;
      } catch (error, stackTrace) {
        _logResultsPage(
          'Failed to resolve image size for $imagePath',
          error: error,
          stackTrace: stackTrace,
        );
        return null;
      }
    }();

    final Future<Size?> tracked = loader.then((Size? value) {
      if (value == null) {
        _imageSizeCache.remove(cacheKey);
      }
      return value;
    });
    _imageSizeCache[cacheKey] = tracked;
    return tracked;
  }

  static Future<Size> _decodeImageSize(ImageProvider provider) {
    final Completer<Size> completer = Completer<Size>();
    final ImageStream stream = provider.resolve(const ImageConfiguration());
    late final ImageStreamListener listener;
    listener = ImageStreamListener(
      (ImageInfo info, bool _) {
        stream.removeListener(listener);
        final ui.Image image = info.image;
        completer.complete(Size(
          image.width.toDouble(),
          image.height.toDouble(),
        ));
      },
      onError: (Object error, StackTrace? stackTrace) {
        stream.removeListener(listener);
        completer.completeError(error, stackTrace ?? StackTrace.current);
      },
    );
    stream.addListener(listener);
    return completer.future;
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
    // First, try to get image dimensions from defectDetection map (most reliable)
    final dynamic detectionMeta = defectDetection?['image_dimensions'] ?? defectDetection?['image_size'];
    double? metaWidth;
    double? metaHeight;
    if (detectionMeta is Map) {
      metaWidth = _asDouble(detectionMeta['width']);
      metaHeight = _asDouble(detectionMeta['height']);
    }
    if (metaWidth != null && metaHeight != null && metaWidth > 0 && metaHeight > 0) {
      return Size(metaWidth, metaHeight);
    }
    
    // Then check individual detections
    for (final detection in detections) {
      if (detection is! Map) continue;
      final map = Map<String, dynamic>.from(detection);

      double? width = _asDouble(map['image_width']) ?? _asDouble(map['imageWidth']);
      double? height = _asDouble(map['image_height']) ?? _asDouble(map['imageHeight']);

      final dynamic imageSizeRaw = map['image_size'] ?? map['imageSize'] ?? map['image_dimensions'];
      if (imageSizeRaw is Map) {
        final dims = Map<String, dynamic>.from(imageSizeRaw);
        width ??= _asDouble(dims['width']);
        height ??= _asDouble(dims['height']);
      }

      if (width != null && height != null && width > 0 && height > 0) {
        return Size(width, height);
      }
    }

    double maxX = 0;
    double maxY = 0;
    bool hasCoordinates = false;
    for (final detection in detections) {
      if (detection is! Map) continue;
      final coords = detection['coordinates'];
      if (coords is! Map) continue;
      final double? x2 = _asDouble(coords['x2']);
      final double? y2 = _asDouble(coords['y2']);
      if (x2 != null) {
        maxX = math.max(maxX, x2);
        hasCoordinates = true;
      }
      if (y2 != null) {
        maxY = math.max(maxY, y2);
        hasCoordinates = true;
      }
    }
    if (!hasCoordinates) {
      return null;
    }
    if (maxX <= 1.0 && maxY <= 1.0) {
      return const Size(1, 1);
    }

    final double largest = math.max(maxX, maxY);
    if (largest > 0 && largest <= 256) {
      return const Size(224, 224);
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
    final Map<String, dynamic>? defectSummary = _getDefectSummary();
    final List<dynamic> detectionsAll = _getDetections();
    final double? predictedDaysRaw = _asDouble(shelfLifeData?['predicted_days']);
    final List<dynamic> detectionsForAdjustment = _filterOutGoodBeanDetections(detectionsAll);
    final int highestRankForShelf = math.max(
      _highestDefectTypeRank(detectionsForAdjustment),
      _highestDefectTypeRankFromMap(defectSummary?['defect_types']),
    );
    final double? weightedDays = _weightedShelfLifeDays(
      typeCounts: defectSummary?['defect_types'] as Map<String, dynamic>?,
      detections: detectionsAll,
    );
    double predictedDays = weightedDays ??
        (predictedDaysRaw != null && predictedDaysRaw > 0 ? predictedDaysRaw : 30.0);
    double? estimatedMonths = double.parse((predictedDays / 30.0).toStringAsFixed(1));
    final Map<String, dynamic>? monthsRange = _normalizeMonthsRange(
      shelfLifeData?['estimated_months_range'] is Map
          ? Map<String, dynamic>.from(shelfLifeData!['estimated_months_range'] as Map)
          : null,
      estimatedMonths,
    );
    String? severityLabel = (defectSummary?['severity'] as String?) ?? (shelfLifeData?['severity'] as String?);
    double defectPct =
        _asDouble(defectSummary?['defect_percentage']) ?? _asDouble(shelfLifeData?['defect_percentage']) ?? 0.0;
    if (defectPct <= 0 && detectionsForAdjustment.isNotEmpty) {
      defectPct = _estimateDefectPercentageFromDetections(detectionsForAdjustment) ?? 0.0;
    }
    final int predictedDaysDisplay =
        (predictedDays ?? _asDouble(shelfLifeData?['predicted_days']) ?? 0).round();
    String shelfLifeCategory = _deriveShelfLifeCategory({'predicted_days': predictedDays});
    final double confidenceScore = shelfLifeData != null
        ? (_asDouble(shelfLifeData['confidence_score'] ?? shelfLifeData['confidence']) ?? 0.0)
        : (healthyPct / 100.0);
    const bool hasGoodBeansOnly = false;
    final Map<int, int> rankCountMap = _rankCounts(
      typeCounts: defectSummary?['defect_types'] as Map<String, dynamic>?,
      detections: detectionsForAdjustment,
    );
    final int highestRankCount = rankCountMap[highestRankForShelf] ?? 0;
    final String defectCategoryLabel = _defectCategoryLabel(
      highestRankForShelf,
      goodBeansOnly: hasGoodBeansOnly,
      count: highestRankCount,
    );
    final List<String> detectionRecommendations = _recommendationsForDetections(detectionsAll);
    final String statusLabel = _resolveStatus(
      category: shelfLifeCategory,
      severity: severityLabel,
      predictedDays: predictedDays,
      baseDays: weightedDays ?? predictedDaysRaw ?? 30.0,
    );
    final onSurface = colorScheme.onSurface;
    final surface = colorScheme.surface;
    final isDark = colorScheme.brightness == Brightness.dark;
    final borderColor = isDark ? colorScheme.outline.withValues(alpha: 0.4) : AppColors.dividerGrey;
    final primaryTextColor = isDark ? colorScheme.onSurface : AppColors.textDarkGrey;
    final dividerColor = isDark ? colorScheme.outline.withValues(alpha: 0.3) : AppColors.dividerGrey;
    final chipBackground = _getShelfLifeColor(
      colorScheme,
      shelfLifeCategory,
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
          if (shelfLife != null || predictedDays != null) ...[
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
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Recommendation:',
                style: TextStyle(fontWeight: FontWeight.w500, color: primaryTextColor.withValues(alpha: 0.75)),
              ),
              Flexible(
                child: Text(
                  detectionRecommendations.isNotEmpty ? 'See numbered list' : defectCategoryLabel,
                  textAlign: TextAlign.right,
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: primaryTextColor,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (detectionRecommendations.isNotEmpty) ...[
            Text(
              'Numbered Recommendations:',
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: primaryTextColor.withValues(alpha: 0.8),
              ),
            ),
            const SizedBox(height: 6),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: detectionRecommendations
                  .map((rec) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text(
                          rec,
                          style: TextStyle(
                            fontWeight: FontWeight.w500,
                            color: primaryTextColor,
                          ),
                        ),
                      ))
                  .toList(),
            ),
            const SizedBox(height: 8),
          ],
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
      case 'normal':
        return 'Normal';
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
    if (percentage <= 0) return 'normal';
    if (percentage < 22) return 'mild';
    if (percentage < 78) return 'moderate';
    return 'severe';
  }

  static const Map<String, int> _baseShelfLifeDays = {
    'arabica': 900, // 30 months baseline
    'liberica': 840, // 28 months baseline
    'excelsa': 780, // 26 months baseline
    'robusta': 750, // 25 months baseline
    'other': 600, // 20 months baseline
  };

  static const Map<String, double> _defectWeights = {
    'fully black': 1.0,
    'fully_black': 1.0,
    'black bean': 1.0,
    'roasted': 0.7,
    'roasted beans': 0.7,
    'insect': 0.5,
    'insect damaged': 0.5,
    'broken': 0.2,
    'broken/cut': 0.2,
    'broken_cut': 0.2,
    'cut': 0.2,
    'good_beans': 0.0,
    'good beans': 0.0,
    'good bean': 0.0,
  };

  static const double _baselineShelfLifeDays = 900.0; // ~30 months midpoint for clean beans
  static const Map<String, double> _defectReductions = {
    'fully_black': 0.75, // 40–60%+ drop -> increased to 75%
    'roasted': 0.833, // target ~5 months (≈150 days) from 900-day baseline
    'insect': 0.73, // target ~8 months (≈240 days) from 900-day baseline
    'broken': 0.25, // 10–25% drop -> use 25%
    'good_beans': 0.0,
  };

  String? _normalizeShelfLifeBucket(String? raw) {
    if (raw == null) return null;
    final normalized = raw
        .toLowerCase()
        .replaceAll(RegExp(r'[_-]+'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    if (normalized.contains('fully black') || normalized.contains('full black') || normalized.contains('black bean')) {
      return 'fully_black';
    }
    if (normalized.contains('roast')) return 'roasted';
    if (normalized.contains('insect')) return 'insect';
    if (normalized.contains('broken') || normalized.contains('cut')) return 'broken';
    if (normalized.contains('good bean')) return 'good_beans';
    return null;
  }

  double? _weightedShelfLifeDays({Map<String, dynamic>? typeCounts, List<dynamic>? detections}) {
    final Map<String, int> counts = {};

    void bump(String? raw, int amount) {
      final bucket = _normalizeShelfLifeBucket(raw);
      if (bucket == null || amount <= 0) return;
      counts[bucket] = (counts[bucket] ?? 0) + amount;
    }

    if (typeCounts != null) {
      typeCounts.forEach((key, value) {
        final int count = (value is num) ? value.toInt() : 0;
        if (count > 0) bump(key?.toString(), count);
      });
    }

    if (detections != null) {
      for (final detection in detections) {
        if (detection is! Map) continue;
        final String? rawType = (detection['defect_type'] ??
                detection['label'] ??
                detection['class'] ??
                detection['type'])
            ?.toString();
        bump(rawType, 1);
      }
    }

    final int totalBeans = counts.values.fold(0, (prev, c) => prev + c);
    if (totalBeans <= 0) return null;
    final int fullyBlackCount = counts['fully_black'] ?? 0;
    if (fullyBlackCount > totalBeans / 2) {
      return 0.0;
    }

    double weightedSum = 0;
    counts.forEach((key, value) {
      final weight = _defectWeights[key] ?? 0.0;
      weightedSum += weight * value;
    });

    double reductionSum = 0;
    counts.forEach((key, value) {
      final reduction = _defectReductions[key] ?? 0.3;
      reductionSum += reduction * value;
    });

    final double avgReduction = (reductionSum / totalBeans).clamp(0.0, 0.9);
    final double days = (_baselineShelfLifeDays * (1.0 - avgReduction)).clamp(0.0, _baselineShelfLifeDays);
    return days;
  }

  int _defectTypeRank(String? defectType) {
    if (defectType == null) return 0;
    final normalized = defectType
        .toLowerCase()
        .replaceAll(RegExp(r'[_-]+'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    if (normalized.contains('fully black') || normalized.contains('full black') || normalized.contains('black bean')) {
      return 4;
    }
    if (normalized.contains('insect')) return 3;
    if (normalized.contains('broken') || normalized.contains('cut')) return 2;
    if (normalized.contains('roast')) return 1;
    return 0;
  }

  int _highestDefectTypeRank(Iterable<dynamic> detections) {
    int highest = 0;
    for (final detection in detections) {
      if (detection is! Map) continue;
      final String? defectType = (detection['defect_type'] ??
              detection['label'] ??
              detection['class'] ??
              detection['type'])
          ?.toString();
      highest = math.max(highest, _defectTypeRank(defectType));
    }
    return highest;
  }

  int _highestDefectTypeRankFromMap(dynamic defectTypes) {
    if (defectTypes is! Map) return 0;
    int highest = 0;
    defectTypes.forEach((key, _) {
      highest = math.max(highest, _defectTypeRank(key?.toString()));
    });
    return highest;
  }

  double _severityFloorPct(int rank) {
    switch (rank) {
      case 4:
        return 85.0;
      case 3:
        return 65.0;
      case 2:
        return 45.0;
      case 1:
        return 20.0;
      default:
        return 0.0;
    }
  }

  String? _severityLabelForRank(int rank) {
    switch (rank) {
      case 4:
        return 'severe';
      case 3:
        return 'moderate';
      case 2:
        return 'mild';
      case 1:
        return 'mild';
      default:
        return null;
    }
  }

  bool _isGoodBeansType(String? defectType) {
    if (defectType == null) return false;
    final normalized = defectType
        .toLowerCase()
        .replaceAll(RegExp(r'[_-]+'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    return normalized.contains('good bean');
  }

  String _normalizeDefectKey(String raw) {
    return raw
        .toLowerCase()
        .replaceAll(RegExp(r'[_-]+'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  double _weightedDefectScore({Map<String, dynamic>? typeCounts, List<dynamic>? detections}) {
    double weightedSum = 0;
    int totalCount = 0;

    if (typeCounts != null) {
      typeCounts.forEach((key, value) {
        final normalizedKey = _normalizeDefectKey(key.toString());
        if (_isGoodBeansType(normalizedKey)) return;
        final int count = (value is num) ? value.toInt() : 0;
        if (count <= 0) return;
        final double weight = _defectWeights[normalizedKey] ?? _defectWeights[normalizedKey.replaceAll(' ', '_')] ?? 0.3;
        weightedSum += count * weight;
        totalCount += count;
      });
    }

    if (detections != null) {
      for (final detection in detections) {
        if (detection is! Map) continue;
        final String? rawType = (detection['defect_type'] ?? detection['label'] ?? detection['class'] ?? detection['type'])
            ?.toString();
        if (rawType == null) continue;
        final normalizedKey = _normalizeDefectKey(rawType);
        if (_isGoodBeansType(normalizedKey)) continue;
        final double weight = _defectWeights[normalizedKey] ?? _defectWeights[normalizedKey.replaceAll(' ', '_')] ?? 0.3;
        weightedSum += weight;
        totalCount += 1;
      }
    }

    if (totalCount == 0) return 0.0;
    final double score = weightedSum / totalCount;
    return score.clamp(0.0, 1.0);
  }

  double _shelfLifeMultiplierFromScore(double score) {
    // Linear decay: at score=0 -> 1.0x, score=1 -> 0.3x (floor 0.25)
    final double linear = 1.0 - (0.7 * score);
    return linear.clamp(0.25, 1.0);
  }

  int _baseShelfLifeDaysForBean(String beanType) {
    final lower = beanType.toLowerCase();
    if (lower.contains('arabica')) return _baseShelfLifeDays['arabica']!;
    if (lower.contains('liberica')) return _baseShelfLifeDays['liberica']!;
    if (lower.contains('excelsa')) return _baseShelfLifeDays['excelsa']!;
    if (lower.contains('robusta')) return _baseShelfLifeDays['robusta']!;
    return _baseShelfLifeDays['other']!;
  }

  int _defectRankForType(String? defectType) {
    return _defectTypeRank(defectType);
  }

  Map<int, int> _rankCounts({
    Map<String, dynamic>? typeCounts,
    List<dynamic>? detections,
  }) {
    final Map<int, int> counts = {};

    if (typeCounts != null) {
      typeCounts.forEach((key, value) {
        final normalizedKey = _normalizeDefectKey(key.toString());
        if (_isGoodBeansType(normalizedKey)) return;
        final int rank = _defectRankForType(normalizedKey);
        final int count = (value is num) ? value.toInt() : 0;
        if (count > 0) {
          counts[rank] = (counts[rank] ?? 0) + count;
        }
      });
    }

    if (detections != null) {
      for (final detection in detections) {
        if (detection is! Map) continue;
        final String? rawType = (detection['defect_type'] ?? detection['label'] ?? detection['class'] ?? detection['type'])
            ?.toString();
        if (rawType == null) continue;
        final normalizedKey = _normalizeDefectKey(rawType);
        if (_isGoodBeansType(normalizedKey)) continue;
        final int rank = _defectRankForType(normalizedKey);
        counts[rank] = (counts[rank] ?? 0) + 1;
      }
    }

    return counts;
  }

  String _recommendationTextForType(String defectType) {
    final normalized = _normalizeDefectKey(defectType);
    if (_isGoodBeansType(normalized)) return 'Store properly and use';
    if (normalized.contains('fully black') || normalized.contains('full black') || normalized.contains('black bean')) {
      return 'Discard immediately';
    }
    if (normalized.contains('insect')) return 'Sort out or reject';
    if (normalized.contains('broken') || normalized.contains('cut')) return 'Use with care; roast separately';
    if (normalized.contains('roast')) return 'Remove before roasting / already roasted';
    return 'Normal';
  }

  List<String> _recommendationsForDetections(List<dynamic> detections) {
    final Map<String, Map<String, dynamic>> recMap = {};
    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];
      if (detection is! Map) continue;
      final String defectType =
          (detection['defect_type'] ?? detection['label'] ?? detection['class'] ?? detection['type'] ?? 'unknown')
              .toString();
      final String normalized = _normalizeDefectKey(defectType);
      final String rec = _recommendationTextForType(defectType);
      recMap.putIfAbsent(rec, () => {'count': 0, 'positions': <int>[]});
      recMap[rec]!['count'] = (recMap[rec]!['count'] as int) + 1;
      (recMap[rec]!['positions'] as List<int>).add(i + 1); // overlay numbering is 1-based index
    }

    int idx = 1;
    final List<String> lines = [];
    recMap.forEach((rec, data) {
      final int count = data['count'] as int;
      final List<int> positions = List<int>.from(data['positions'] as List);
      positions.sort();
      final String posLabel = positions.isNotEmpty ? 'items: ${positions.join(', ')}' : '';
      lines.add('${idx++}: $rec (count: $count${posLabel.isNotEmpty ? ', $posLabel' : ''})');
    });
    return lines;
  }

  int _defectDisplayNumber(int rank, {bool goodBeansOnly = false}) {
    if (goodBeansOnly) return 5;
    switch (rank) {
      case 4:
        return 1;
      case 3:
        return 2;
      case 2:
        return 3;
      case 1:
        return 4;
      default:
        return 0;
    }
  }

  String _defectCategoryLabel(int rank, {bool goodBeansOnly = false, int count = 0}) {
    final int displayNumber = _defectDisplayNumber(rank, goodBeansOnly: goodBeansOnly);
    String recommendation;
    if (goodBeansOnly) {
      recommendation = 'Store properly and use';
    } else {
      switch (rank) {
        case 4:
          recommendation = 'Discard immediately';
          break;
        case 3:
          recommendation = 'Sort out or reject';
          break;
        case 2:
          recommendation = 'Use with care; roast separately';
          break;
        case 1:
          recommendation = 'Remove before roasting / already roasted';
          break;
        default:
          recommendation = 'Normal';
          break;
      }
    }
    final String countSuffix = count > 0 ? ' (count: $count)' : '';
    return displayNumber > 0 ? '$displayNumber: $recommendation$countSuffix' : '$recommendation$countSuffix';
  }

  bool _hasGoodBeans(Iterable<dynamic> detections) {
    for (final detection in detections) {
      if (detection is! Map) continue;
      final String? defectType = (detection['defect_type'] ??
              detection['label'] ??
              detection['class'] ??
              detection['type'])
          ?.toString();
      if (_isGoodBeansType(defectType)) {
        return true;
      }
    }
    return false;
  }

  List<dynamic> _filterOutGoodBeanDetections(Iterable<dynamic> detections) {
    return detections
        .where((detection) {
          if (detection is! Map) return true;
          final String? defectType = (detection['defect_type'] ??
                  detection['label'] ??
                  detection['class'] ??
                  detection['type'])
              ?.toString();
          return !_isGoodBeansType(defectType);
        })
        .toList();
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

  String _resolveStatus({String? category, String? severity, num? predictedDays, num? baseDays}) {
    String? normalizedSeverity = _normalizeSeverityTag(severity);
    final String? rawCategory = category?.trim().isNotEmpty == true ? category!.trim() : null;
    final String? normalizedCategory = _normalizeSeverityTag(rawCategory);

    if ((normalizedSeverity == null || normalizedSeverity.isEmpty) && predictedDays != null) {
      final double anchor = (baseDays != null && baseDays > 0)
          ? baseDays.toDouble()
          : 360.0; // default 12 months when anchor missing
      final double ratio = (predictedDays.toDouble() / anchor).clamp(0.0, 1.5);
      if (ratio >= 0.9) {
        normalizedSeverity = 'mild';
      } else if (ratio >= 0.6) {
        normalizedSeverity = 'moderate';
      } else {
        normalizedSeverity = 'severe';
      }
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
      case 'normal':
        return 'normal';
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
      case 'normal':
        return 'Normal';
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
    final List<dynamic> rawDetections = _getDetections();
    final List<dynamic> detections = _filterOutGoodBeanDetections(rawDetections);
    final bool hasGoodBeans = _hasGoodBeans(rawDetections);
    final double? shelfLifePct =
        _asDouble(shelfLife?['defect_percentage'] ?? shelfLife?['defective_percent']);
    final int rawTotalDefects =
        summary?['total_defects'] is num ? (summary!['total_defects'] as num).toInt() : detections.length;
    final double? summaryPct = _asDouble(summary?['defect_percentage']);

    double? resolvedPctCandidate = shelfLifePct;
    if (resolvedPctCandidate == null || (resolvedPctCandidate <= 0 && summaryPct != null && summaryPct > 0)) {
      resolvedPctCandidate = summaryPct;
    }

    final bool hasDetectedDefects = rawTotalDefects > 0 || detections.isNotEmpty;
    final bool hasGoodBeansOnly = hasGoodBeans && !hasDetectedDefects;
    final int totalDefects = hasGoodBeansOnly ? 0 : rawTotalDefects;

    double resolvedPctValue;
    final int highestRank = math.max(
      _highestDefectTypeRank(detections),
      _highestDefectTypeRankFromMap(summary?['defect_types']),
    );
    if (hasGoodBeansOnly) {
      resolvedPctValue = 0.0;
    } else if (totalDefects > 0) {
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

    resolvedPctValue = math.max(resolvedPctValue, _severityFloorPct(highestRank));
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

    final String? rankSeverity = _severityLabelForRank(highestRank);
    if (hasDetectedDefects &&
        rankSeverity != null &&
        _severityRank(rankSeverity) > _severityRank(severityLabel)) {
      severityLabel = rankSeverity;
    }

    if (!hasDetectedDefects && !hasGoodBeansOnly) {
      severityLabel = 'normal';
    }
    if (hasGoodBeansOnly) {
      severityLabel = 'normal';
    }

    int severityLevel;
    switch (severityLabel.toLowerCase()) {
      case 'normal':
        severityLevel = 1;
        break;
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
    int highestRank = 0;
    bool hasGoodBeans = _hasGoodBeans(_getDetections());
    if (defectDetection?['summary'] is Map<String, dynamic>) {
      summary = Map<String, dynamic>.from(defectDetection!['summary'] as Map);
      highestRank = math.max(highestRank, _highestDefectTypeRankFromMap(summary['defect_types']));
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

    int removedGoodBeanCounts = 0;
    if (summary['defect_types'] is Map) {
      final Map<String, dynamic> types = Map<String, dynamic>.from(summary['defect_types'] as Map);
      final Map<String, dynamic> filtered = {};
      types.forEach((key, value) {
        if (_isGoodBeansType(key.toString())) {
          if (value is num) removedGoodBeanCounts += value.toInt();
          return;
        }
        filtered[key] = value;
      });
      summary['defect_types'] = filtered;
      if (removedGoodBeanCounts > 0) {
        hasGoodBeans = true;
      }
      if (removedGoodBeanCounts > 0) {
        final int currentTotal = _asInt(summary['total_defects']) ?? 0;
        summary['total_defects'] = math.max(0, currentTotal - removedGoodBeanCounts);
      }
      highestRank = math.max(highestRank, _highestDefectTypeRankFromMap(filtered));
    }

    final Map<String, dynamic>? shelfLifeData =
        shelfLife != null ? Map<String, dynamic>.from(shelfLife!) : null;
    final double? shelfLifePct =
        _asDouble(shelfLifeData?['defect_percentage'] ?? shelfLifeData?['defective_percent']);

    final detections = _getDetectionsForSummary();
    highestRank = math.max(highestRank, _highestDefectTypeRank(detections));
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
        if (_isGoodBeansType(key)) continue;
        final value = entry.value;
        if (value is num) {
          final current = typeCounts[key] ?? 0;
          typeCounts[key] = math.max(current, value.toInt());
        }
      }
    }
    highestRank = math.max(highestRank, _highestDefectTypeRankFromMap(typeCounts));
    if (typeCounts.isNotEmpty) {
      summary['defect_types'] =
          typeCounts.map((key, value) => MapEntry(_formatDefectType(key), value));
      final int countsTotal = typeCounts.values.fold(0, (prev, value) => prev + value);
      summary['total_defects'] = math.max(_asInt(summary['total_defects']) ?? 0, countsTotal);
    }

    final Map<String, dynamic>? summaryDefectTypes =
        summary['defect_types'] is Map ? Map<String, dynamic>.from(summary['defect_types'] as Map) : null;
    final bool hasDefectTypes = summaryDefectTypes != null && summaryDefectTypes.isNotEmpty;
    if (hasGoodBeans && !hasDefectTypes && detections.isEmpty) {
      summary['total_defects'] = 0;
      summary['defect_percentage'] = 0.0;
      summary['severity'] = 'normal';
      summary['quality_grade'] ??= 'A';
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

    if (hasGoodBeans && (_asInt(summary['total_defects']) ?? 0) == 0 && detections.isEmpty) {
      summaryPctValue = 0.0;
    }

    summaryPctValue = math.max(summaryPctValue, _severityFloorPct(highestRank));
    final double cappedPct = summaryPctValue.clamp(0.0, 100.0);
    summary['defect_percentage'] = cappedPct;
    String normalizedSeverity =
        _normalizeSeverityTag(severity) ?? _computeSeverityFromPercentage(cappedPct);
    final String? rankSeverity = _severityLabelForRank(highestRank);
    if (rankSeverity != null && _severityRank(rankSeverity) > _severityRank(normalizedSeverity)) {
      normalizedSeverity = rankSeverity;
    }
    summary['severity'] = normalizedSeverity;

    final int normalizedTotalDefects = _asInt(summary['total_defects']) ?? detections.length;
    if (normalizedTotalDefects <= 0 && detections.isEmpty) {
      if (hasGoodBeans) {
        summary['severity'] = 'mild';
        summary['defect_percentage'] = 0.0;
      } else {
        summary['severity'] = 'normal';
      }
    }

    return summary;
  }

  List<dynamic> _getDetections() {
    if (defectDetection == null) return const [];
    final dynamic preferredList =
        defectDetection?['detections_all'] ?? defectDetection?['detections'];
    return _normalizeDetectionList(preferredList ?? defectDetection);
  }

  List<dynamic> _getDetectionsForSummary() {
    if (defectDetection == null) return const [];
    final dynamic filteredList = defectDetection?['detections'];
    final normalized = _normalizeDetectionList(filteredList ?? defectDetection);
    return _filterOutGoodBeanDetections(normalized);
  }

  List<Map<String, dynamic>> _normalizeDetectionList(dynamic listData) {
    Map<String, dynamic>? normalize(dynamic raw) {
      if (raw is! Map) return null;
      final source = Map<String, dynamic>.from(raw);

      final coords = _extractCoordinateMap(source);
      if (coords == null) {
        _logResultsPage('Detection entry missing coordinate information: $source');
        return null;
      }

      final double x1 = _parseCoordinate(coords['x1'] ?? coords['left'] ?? coords['xmin'] ?? coords['x']);
      final double y1 = _parseCoordinate(coords['y1'] ?? coords['top'] ?? coords['ymin'] ?? coords['y']);
      double x2 = _parseCoordinate(coords['x2'] ?? coords['right'] ?? coords['xmax']);
      double y2 = _parseCoordinate(coords['y2'] ?? coords['bottom'] ?? coords['ymax']);
      final double width = _parseCoordinate(coords['width'] ?? coords['w']);
      final double height = _parseCoordinate(coords['height'] ?? coords['h']);

      if ((x2 <= x1 || !x2.isFinite) && width > 0) {
        x2 = x1 + width;
      }
      if ((y2 <= y1 || !y2.isFinite) && height > 0) {
        y2 = y1 + height;
      }

      final double normalizedConfidence = _normalizeConfidenceValue(
        source['confidence'] ?? source['score'] ?? source['probability'],
      );

      final String defectType = (source['defect_type'] ??
              source['label'] ??
              source['class'] ??
              source['type'] ??
              'unknown')
          .toString();

      final Map<String, dynamic> normalized = {
        ...source,
        'defect_type': defectType,
        'confidence': normalizedConfidence,
        'coordinates': {
          'x1': x1,
          'y1': y1,
          'x2': x2,
          'y2': y2,
        },
      };

      final double? imageWidth = _asDouble(source['image_width']) ??
          _asDouble(source['imageWidth']) ??
          _asDouble((source['image_size'] as Map?)?['width']) ??
          _asDouble((source['imageSize'] as Map?)?['width']);
      final double? imageHeight = _asDouble(source['image_height']) ??
          _asDouble(source['imageHeight']) ??
          _asDouble((source['image_size'] as Map?)?['height']) ??
          _asDouble((source['imageSize'] as Map?)?['height']);

      if (imageWidth != null) {
        normalized['image_width'] = imageWidth;
      }
      if (imageHeight != null) {
        normalized['image_height'] = imageHeight;
      }
      if (imageWidth != null || imageHeight != null) {
        normalized['image_size'] = {
          'width': imageWidth,
          'height': imageHeight,
        };
      } else {
        // If detection doesn't have image dimensions, try to get from defectDetection map
        final dynamic detectionMeta = defectDetection?['image_dimensions'] ?? defectDetection?['image_size'];
        if (detectionMeta is Map) {
          final dimsMap = Map<String, dynamic>.from(detectionMeta);
          final double? metaWidth = _asDouble(dimsMap['width']);
          final double? metaHeight = _asDouble(dimsMap['height']);
          if (metaWidth != null && metaHeight != null && metaWidth > 0 && metaHeight > 0) {
            normalized['image_width'] = metaWidth;
            normalized['image_height'] = metaHeight;
            normalized['image_size'] = {
              'width': metaWidth,
              'height': metaHeight,
            };
          }
        }
      }

      return normalized;
    }

    if (listData is List) {
      return listData.map<Map<String, dynamic>?>(normalize).whereType<Map<String, dynamic>>().toList();
    }

    final Map<String, dynamic>? single = normalize(defectDetection!);
    return single == null ? <Map<String, dynamic>>[] : [single];
  }

  Map<String, dynamic>? _extractCoordinateMap(Map<String, dynamic> source) {
    final dynamic direct = source['coordinates'] ?? source['defect_coordinates'];
    if (direct is Map) {
      return Map<String, dynamic>.from(direct);
    }

    final dynamic bbox = source['bbox'] ?? source['box'];
    if (bbox is List && bbox.length >= 4) {
      return {
        'x1': bbox[0],
        'y1': bbox[1],
        'x2': bbox[2],
        'y2': bbox[3],
      };
    }
    if (bbox is Map) {
      return Map<String, dynamic>.from(bbox);
    }

    if (source.containsKey('x') && source.containsKey('y')) {
      return {
        'x1': source['x'],
        'y1': source['y'],
        'width': source['width'] ?? source['w'],
        'height': source['height'] ?? source['h'],
      };
    }

    if (source.containsKey('left') && source.containsKey('top')) {
      return {
        'x1': source['left'],
        'y1': source['top'],
        'x2': source['right'],
        'y2': source['bottom'],
        'width': source['width'] ?? source['w'],
        'height': source['height'] ?? source['h'],
      };
    }

    return null;
  }

  double _parseCoordinate(dynamic value) {
    final parsed = _asDouble(value);
    if (parsed == null || parsed.isNaN || !parsed.isFinite) {
      return 0.0;
    }
    return parsed;
  }

  double _normalizeConfidenceValue(dynamic value) {
    final double raw = _asDouble(value) ?? 0.0;
    if (raw > 1.0) {
      return (raw / 100.0).clamp(0.0, 1.0);
    }
    if (raw < 0.0) {
      return 0.0;
    }
    return raw;
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
    final double withRankFloor =
        math.max(estimated, _severityFloorPct(_highestDefectTypeRank(detections)));
    return withRankFloor.clamp(24.0, 95.0);
  }

  // Derive a readable shelf-life status if backend did not store the category
  String _deriveShelfLifeCategory(Map<String, dynamic> shelf) {
    final int days = (shelf['predicted_days'] as num?)?.toInt() ?? 0;
    if (days >= 30) return 'Excellent';
    if (days >= 20) return 'Good';
    if (days >= 10) return 'Warning';
    if (days > 0) return 'Critical';
    return 'Critical';
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

  Widget _buildActionButtons(BuildContext context, ColorScheme colorScheme) {
    final textColor = colorScheme.onSurface.withValues(alpha: 0.9);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          'Scan another image?',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: textColor,
          ),
        ),
        const SizedBox(height: AppConstants.mediumSpacing),
        Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () => Navigator.of(context).pop(ResultsNavigationAction.scan),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size.fromHeight(48),
                  textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                ),
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Yes'),
              ),
            ),
            const SizedBox(width: AppConstants.mediumSpacing),
            Expanded(
              child: OutlinedButton(
                onPressed: () => Navigator.of(context).pop(ResultsNavigationAction.history),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size.fromHeight(48),
                  textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                ),
                child: const Text('No'),
              ),
            ),
          ],
        ),
      ],
    );
  }

}

class DefectAnnotationPainter extends CustomPainter {
  final List<dynamic> detections;
  final Size? originalSize;
  final Size? actualDisplayedSize; // Actual displayed image size (after EXIF)
  final Size? imageDisplayedSize; // The calculated displayed image size (for coordinate matching)
  final Offset imageOffset; // The offset where the image is positioned
  
  DefectAnnotationPainter(
    this.detections, {
    this.originalSize,
    this.actualDisplayedSize,
    this.imageDisplayedSize,
    Offset? imageOffset,
  }) : imageOffset = imageOffset ?? Offset.zero;

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
    final List<Map<String, double>?> coordsByDetection =
        List<Map<String, double>?>.filled(detections.length, null, growable: false);
    double minX = double.infinity;
    double minY = double.infinity;
    double maxX = -double.infinity;
    double maxY = -double.infinity;
    bool hasAnyCoordinates = false;

    for (int i = 0; i < detections.length; i++) {
      final coords = _resolveCoordinates(detections[i]);
      coordsByDetection[i] = coords;
      if (coords == null) {
        continue;
      }
      hasAnyCoordinates = true;
      final double localMinX = math.min(coords['x1']!, coords['x2']!);
      final double localMinY = math.min(coords['y1']!, coords['y2']!);
      final double localMaxX = math.max(coords['x1']!, coords['x2']!);
      final double localMaxY = math.max(coords['y1']!, coords['y2']!);
      minX = math.min(minX, localMinX);
      minY = math.min(minY, localMinY);
      maxX = math.max(maxX, localMaxX);
      maxY = math.max(maxY, localMaxY);
    }

    if (!hasAnyCoordinates) {
      _logResultsPage('No detections with valid coordinates available for painting');
      return;
    }

    if (!minX.isFinite) minX = 0;
    if (!minY.isFinite) minY = 0;
    if (!maxX.isFinite) maxX = 0;
    if (!maxY.isFinite) maxY = 0;

    // Get original image dimensions from detections or originalSize parameter
    final bool hasOriginalSize = originalSize != null && originalSize!.width > 0 && originalSize!.height > 0;
    final bool appearsNormalized = maxX <= 1.0 && maxY <= 1.0 && minX >= 0 && minY >= 0;

    // Initialize with default values to ensure they're always assigned
    double sourceWidth = 1.0;
    double sourceHeight = 1.0;

    if (hasOriginalSize) {
      sourceWidth = originalSize!.width;
      sourceHeight = originalSize!.height;
    } else if (appearsNormalized) {
      sourceWidth = 1.0;
      sourceHeight = 1.0;
    } else {
      // Try to get image dimensions from first detection
      bool foundImageSize = false;
      for (final detection in detections) {
        if (detection is Map) {
          final double? imgWidth = _parseDouble(detection['image_width'] ?? detection['imageWidth'] ?? (detection['image_size'] as Map?)?['width'] ?? (detection['imageSize'] as Map?)?['width']);
          final double? imgHeight = _parseDouble(detection['image_height'] ?? detection['imageHeight'] ?? (detection['image_size'] as Map?)?['height'] ?? (detection['imageSize'] as Map?)?['height']);
          if (imgWidth != null && imgHeight != null && imgWidth > 0 && imgHeight > 0) {
            sourceWidth = imgWidth;
            sourceHeight = imgHeight;
            foundImageSize = true;
            break;
          }
        }
      }
      if (!foundImageSize) {
        // Fallback: use max coordinates as image size
        sourceWidth = math.max(maxX - minX, 1.0);
        sourceHeight = math.max(maxY - minY, 1.0);
    }
    }

    // Ensure values are valid
    if (sourceWidth <= 0) sourceWidth = 1.0;
    if (sourceHeight <= 0) sourceHeight = 1.0;

    // Check for EXIF orientation mismatch (dimensions swapped)
    // If backend processed image as WxH but Flutter displays as HxW, we need to rotate coordinates
    bool isRotated = false;
    double effectiveSourceWidth = sourceWidth;
    double effectiveSourceHeight = sourceHeight;
    
    if (actualDisplayedSize != null && hasOriginalSize) {
      // Check if dimensions are swapped (indicates 90-degree rotation)
      final bool widthSwapped = (sourceWidth - actualDisplayedSize!.height).abs() < 1.0;
      final bool heightSwapped = (sourceHeight - actualDisplayedSize!.width).abs() < 1.0;
      
      if (widthSwapped && heightSwapped) {
        isRotated = true;
        // Use displayed dimensions for coordinate space (they match what Flutter shows)
        effectiveSourceWidth = actualDisplayedSize!.width;
        effectiveSourceHeight = actualDisplayedSize!.height;
        debugPrint('[ResultsPage] EXIF rotation detected! Backend: ${sourceWidth}x${sourceHeight}, Displayed: ${actualDisplayedSize!.width}x${actualDisplayedSize!.height}. Rotating coordinates.');
        debugPrint('[ResultsPage] Rotation details: widthSwapped=$widthSwapped, heightSwapped=$heightSwapped');
      } else {
        // No rotation - dimensions match
        debugPrint('[ResultsPage] No rotation detected. Backend: ${sourceWidth}x${sourceHeight}, Displayed: ${actualDisplayedSize!.width}x${actualDisplayedSize!.height}');
      }
    }

    // Use the provided imageDisplayedSize and imageOffset if available (from parent widget)
    // This ensures we use the exact same calculations as the image positioning
    double displayedWidth;
    double displayedHeight;
    double offsetX = 0.0;
    double offsetY = 0.0;
    
    if (imageDisplayedSize != null) {
      // Use the provided displayed size and offset (matches image widget positioning)
      displayedWidth = imageDisplayedSize!.width;
      displayedHeight = imageDisplayedSize!.height;
      offsetX = imageOffset.dx;
      offsetY = imageOffset.dy;
      debugPrint('[ResultsPage] Using provided imageDisplayedSize: ${displayedWidth}x${displayedHeight}, offset: ($offsetX, $offsetY)');
    } else {
      // Fallback: Calculate the actual displayed image size considering BoxFit.contain
      // Use effective dimensions (accounting for rotation) for aspect ratio calculation
      final double imageAspectRatio = effectiveSourceWidth / effectiveSourceHeight;
      final double containerAspectRatio = size.width / size.height;
      
      if (imageAspectRatio > containerAspectRatio) {
        // Image is wider than container - fit to width
        displayedWidth = size.width;
        displayedHeight = size.width / imageAspectRatio;
        offsetY = (size.height - displayedHeight) / 2.0; // Center vertically
      } else {
        // Image is taller than container - fit to height
        displayedHeight = size.height;
        displayedWidth = size.height * imageAspectRatio;
        offsetX = (size.width - displayedWidth) / 2.0; // Center horizontally
      }
    }

    // Calculate scale factors: from effective source image size to displayed image size
    final double scaleX = displayedWidth / effectiveSourceWidth;
    final double scaleY = displayedHeight / effectiveSourceHeight;

    final logMsg = 'Coordinate scaling -> original:(${sourceWidth.toStringAsFixed(2)}, ${sourceHeight.toStringAsFixed(2)}) '
      'effective:(${effectiveSourceWidth.toStringAsFixed(2)}, ${effectiveSourceHeight.toStringAsFixed(2)}) '
      'displayed:(${displayedWidth.toStringAsFixed(2)}, ${displayedHeight.toStringAsFixed(2)}) '
      'container:(${size.width.toStringAsFixed(2)}, ${size.height.toStringAsFixed(2)}) '
      'offset:(${offsetX.toStringAsFixed(2)}, ${offsetY.toStringAsFixed(2)}) '
      'scaleX=$scaleX scaleY=$scaleY isRotated=$isRotated';
    _logResultsPage(logMsg);
    debugPrint('[ResultsPage] $logMsg'); // Also print to terminal

    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];
      final coords = coordsByDetection[i];
      if (coords == null) {
        _logResultsPage('Detection $i: No usable coordinates found');
        continue;
      }

      final double x1 = coords['x1']!;
      final double y1 = coords['y1']!;
      final double x2 = coords['x2']!;
      final double y2 = coords['y2']!;
      final String defectType =
          (detection['defect_type'] ?? detection['label'] ?? 'Unknown').toString();
      final double confidence = _normalizeConfidence(
        detection['confidence'] ?? detection['score'] ?? detection['probability'],
      );

      _logResultsPage('Detection $i: $defectType at ($x1, $y1, $x2, $y2) with confidence $confidence');

      // Check if coordinates are valid (not all zeros)
      if ((x1 == 0.0 && y1 == 0.0 && x2 == 0.0 && y2 == 0.0) || x2 <= x1 || y2 <= y1) {
        _logResultsPage('Detection $i: Invalid coordinates, skipping visual overlay but keeping in counts.');
        continue;
      }

      // CRITICAL: Coordinates from backend are in backend's coordinate space
      // We need to map them to the displayed image coordinate space
      // If there's a rotation (dimensions swapped), we must transform coordinates
      
      double coordX1 = x1;
      double coordY1 = y1;
      double coordX2 = x2;
      double coordY2 = y2;
      double coordSourceWidth = sourceWidth;
      double coordSourceHeight = sourceHeight;
      
      if (isRotated) {
        // Backend processed as WxH (e.g., 640x480), Flutter displays as HxW (e.g., 480x640)
        // Try 90° counter-clockwise rotation first (most common for EXIF): (x, y) -> (H - y, x)
        final double backendH = sourceHeight;
        
        // Transform coordinates using counter-clockwise rotation
        // Original: (x1,y1) to (x2,y2) in WxH space
        // After 90° CCW: (H-y1,x1) to (H-y2,x2) in HxW space
        final double ccwX1 = backendH - y2;
        final double ccwX2 = backendH - y1;
        final double ccwY1 = x1;
        final double ccwY2 = x2;
        
        coordX1 = math.min(ccwX1, ccwX2);
        coordX2 = math.max(ccwX1, ccwX2);
        coordY1 = math.min(ccwY1, ccwY2);
        coordY2 = math.max(ccwY1, ccwY2);
        
        // After rotation, coordinate space matches displayed dimensions (HxW)
        coordSourceWidth = effectiveSourceWidth;
        coordSourceHeight = effectiveSourceHeight;
        
        debugPrint('[ResultsPage] Detection $i rotated (CCW): backend=($x1,$y1,$x2,$y2) in ${sourceWidth}x${sourceHeight} -> '
          'transformed=($coordX1,$coordY1,$coordX2,$coordY2) in ${coordSourceWidth}x${coordSourceHeight}');
      }
      
      // Scale coordinates from coordinate source size to displayed image size
      // Then add offset to account for centering (BoxFit.contain)
      final double coordScaleX = displayedWidth / coordSourceWidth;
      final double coordScaleY = displayedHeight / coordSourceHeight;
      
      final scaledX1 = (coordX1 * coordScaleX) + offsetX;
      final scaledY1 = (coordY1 * coordScaleY) + offsetY;
      final scaledX2 = (coordX2 * coordScaleX) + offsetX;
      final scaledY2 = (coordY2 * coordScaleY) + offsetY;
      
      final scalingLogMsg = 'Detection $i scaling: original=($x1,$y1,$x2,$y2) '
        'coordScale=($coordScaleX,$coordScaleY) offset=($offsetX,$offsetY) '
        'scaled=($scaledX1,$scaledY1,$scaledX2,$scaledY2)';
      _logResultsPage(scalingLogMsg);
      debugPrint('[ResultsPage] $scalingLogMsg'); // Also print to terminal
      
      // Clamp scaled coordinates to canvas bounds
      final clampedX1 = scaledX1.clamp(0.0, size.width);
      final clampedY1 = scaledY1.clamp(0.0, size.height);
      final clampedX2 = scaledX2.clamp(0.0, size.width);
      final clampedY2 = scaledY2.clamp(0.0, size.height);
      
      // Check if bounding box is too small to be visible (after scaling)
      final boxWidth = clampedX2 - clampedX1;
      final boxHeight = clampedY2 - clampedY1;
      if (boxWidth < 1.5 || boxHeight < 1.5) {
        _logResultsPage('Detection $i: Bounding box very small after scaling ($boxWidth x $boxHeight), drawing indicator dot instead');
        final dotPaint = Paint()
          ..color = paint.color
          ..style = PaintingStyle.fill;
        final dotCenter = Offset(clampedX1.clamp(4.0, size.width - 4.0), clampedY1.clamp(4.0, size.height - 4.0));
        canvas.drawCircle(dotCenter, 4, dotPaint);

        // Removed numeric badge near tiny detections to avoid duplication

        final labelText =
            '${i + 1}. $defectType (${(confidence * 100).clamp(0, 100).toInt()}%)';
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
      
      _logResultsPage('Detection $i: Scaled from ($x1, $y1, $x2, $y2) to ($clampedX1, $clampedY1, $clampedX2, $clampedY2)');

      // Draw bounding box (outline only) using scaled coordinates
      final rect = Rect.fromLTRB(clampedX1, clampedY1, clampedX2, clampedY2);
      canvas.drawRect(rect, paint);
      validBoxesDrawn++;

      // Removed numeric badge on boxes to avoid double counting visuals

      // Re-layout with descriptive label that mirrors the results list numbering
      final labelText =
          '${i + 1}. $defectType (${(confidence * 100).clamp(0, 100).toInt()}%)';
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

      // Draw label background with dark semi-transparent background
      final labelPaint = Paint()
        ..color = Colors.black.withValues(alpha: 0.6)
        ..style = PaintingStyle.fill;

      canvas.drawRect(labelRect, labelPaint);
      textPainter.paint(canvas, Offset(clampedX1 + 4, labelRect.top + 2));
    }

    // If no coordinates were available but detections exist (e.g., classifier-only defects),
    // draw a single red circle in the center of the displayed image to signal a defect.
    if (!hasAnyCoordinates && detections.isNotEmpty) {
      final Offset center = Offset(offsetX + displayedWidth / 2, offsetY + displayedHeight / 2);
      final double radius = math.min(displayedWidth, displayedHeight) * 0.12;

      final Paint circlePaint = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;

      canvas.drawCircle(center, radius, circlePaint);
      _logResultsPage('Fallback defect indicator drawn at center due to missing coordinates');
    }
    
    _logResultsPage('Valid bounding boxes drawn: $validBoxesDrawn out of ${detections.length} total detections');
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;

  static Map<String, double>? _resolveCoordinates(dynamic detection) {
    if (detection is! Map) return null;
    final Map<dynamic, dynamic> map = detection;
    final dynamic rawCoords = map['coordinates'] ?? map['defect_coordinates'];

    Map<String, dynamic>? coords;
    if (rawCoords is Map) {
      coords = Map<String, dynamic>.from(rawCoords);
    } else {
      final dynamic bbox = map['bbox'] ?? map['box'];
      if (bbox is List && bbox.length >= 4) {
        coords = {
          'x1': bbox[0],
          'y1': bbox[1],
          'x2': bbox[2],
          'y2': bbox[3],
        };
      } else if (bbox is Map) {
        coords = Map<String, dynamic>.from(bbox);
      } else if (map.containsKey('x') && map.containsKey('y')) {
        coords = {
          'x1': map['x'],
          'y1': map['y'],
          'width': map['width'] ?? map['w'],
          'height': map['height'] ?? map['h'],
        };
      } else if (map.containsKey('left') && map.containsKey('top')) {
        coords = {
          'x1': map['left'],
          'y1': map['top'],
          'x2': map['right'],
          'y2': map['bottom'],
          'width': map['width'] ?? map['w'],
          'height': map['height'] ?? map['h'],
        };
      }
    }

    if (coords == null) return null;

    final double x1 =
        _parseDouble(coords['x1'] ?? coords['left'] ?? coords['xmin'] ?? coords['x']);
    final double y1 =
        _parseDouble(coords['y1'] ?? coords['top'] ?? coords['ymin'] ?? coords['y']);
    double x2 = _parseDouble(coords['x2'] ?? coords['right'] ?? coords['xmax']);
    double y2 = _parseDouble(coords['y2'] ?? coords['bottom'] ?? coords['ymax']);
    final double width = _parseDouble(coords['width'] ?? coords['w']);
    final double height = _parseDouble(coords['height'] ?? coords['h']);

    if ((x2 <= x1 || !x2.isFinite) && width > 0) {
      x2 = x1 + width;
    }
    if ((y2 <= y1 || !y2.isFinite) && height > 0) {
      y2 = y1 + height;
    }

    return {
      'x1': x1,
      'y1': y1,
      'x2': x2,
      'y2': y2,
    };
  }

  static double _parseDouble(dynamic value, [double fallback = 0.0]) {
    if (value == null) return fallback;
    if (value is num) return value.toDouble();
    if (value is String) {
      final trimmed = value.trim();
      if (trimmed.isEmpty) return fallback;
      final cleaned = trimmed.replaceAll(RegExp(r'[^0-9\.\-]'), '');
      if (cleaned.isEmpty) return fallback;
      return double.tryParse(cleaned) ?? fallback;
    }
    return fallback;
  }

  static double _normalizeConfidence(dynamic value) {
    final double parsed = _parseDouble(value);
    if (parsed > 1.0) {
      return (parsed / 100.0).clamp(0.0, 1.0);
    }
    if (parsed < 0.0) {
      return 0.0;
    }
    return parsed;
  }
}
