import 'dart:math' as math;

import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';
import '../utils/local_history_store.dart';
import 'results_page.dart';

class HistoryPageController {
  _HistoryPageState? _state;

  void _attach(_HistoryPageState state) {
    _state = state;
  }

  void _detach(_HistoryPageState state) {
    if (identical(_state, state)) {
      _state = null;
    }
  }

  Future<void> refresh({bool showLoadingIndicator = false}) {
    final state = _state;
    if (state == null) {
      return Future.value();
    }
    return state._loadHistory(showLoadingIndicator: showLoadingIndicator);
  }
}

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key, this.controller});

  final HistoryPageController? controller;

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  List<dynamic> _items = [];
  bool _loading = true;

  static const Color _lightBackground = Color(0xFFF0F0F0);
  static const Color _lightCard = Color(0xFFF1E7D2);
  static const Color _lightBorder = Color(0x1A55351C);
  static const Color _lightHeaderText = Color(0xFF554848);
  static const Color _lightBodyText = Color(0xFF554848);
  static const Color _lightIconBackground = Color(0xFFEAD7BC);

  static const Color _darkBackground = Color(0xFF212121);
  static const Color _darkCard = Color(0xFF2A231A);
  static const Color _darkBorder = Color(0xFF3B3329);
  static const Color _darkIconBackground = Color(0xFF463422);

  @override
  void initState() {
    super.initState();
    widget.controller?._attach(this);
    _loadHistory();
  }

  @override
  void didUpdateWidget(covariant HistoryPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!identical(oldWidget.controller, widget.controller)) {
      oldWidget.controller?._detach(this);
      widget.controller?._attach(this);
    }
  }

  @override
  void dispose() {
    widget.controller?._detach(this);
    super.dispose();
  }

  Future<void> _loadHistory({bool showLoadingIndicator = false}) async {
    if (showLoadingIndicator && mounted) {
      setState(() {
        _loading = true;
      });
    }
    final res = await ApiService.fetchHistory(limit: 50);
    if (!mounted) return;

    List<Map<String, dynamic>> combined = [];
    if (res['success'] == true) {
      final data = res['data'] as Map<String, dynamic>? ?? {};
      final scans = data['scans'];
      if (scans is List) {
        combined = scans
            .whereType<Map<String, dynamic>>()
            .map((e) => Map<String, dynamic>.from(e))
            .toList();
      }
    } else {
      debugPrint('History load failed: ${res['error']}');
    }

    final localEntries = await LocalHistoryStore.getEntries();
    final Set<int> historyIds = {};
    combined = combined.map((item) {
      final map = Map<String, dynamic>.from(item);
      final id = map['history_id'];
      if (id is int) historyIds.add(id);
      return map;
    }).toList();

    for (final entry in localEntries) {
      final map = Map<String, dynamic>.from(entry);
      final id = map['history_id'];
      if (id is int && historyIds.contains(id)) {
        continue;
      }
      combined.add(map);
    }

    combined.sort((a, b) {
      DateTime parseDate(dynamic value) {
        if (value is String) {
          return DateTime.tryParse(value) ?? DateTime.now();
        }
        return DateTime.now();
      }

      final aDate = parseDate(a['created_at']);
      final bDate = parseDate(b['created_at']);
      return bDate.compareTo(aDate);
    });

    // Debug: log a few merged items to verify shelf-life/defect fields
    for (int i = 0; i < combined.length && i < 5; i++) {
      final item = combined[i];
      try {
        final Map<String, dynamic> shelfLifeDebug =
            _deriveShelfLifeMap(Map<String, dynamic>.from(item)) ?? <String, dynamic>{};
        debugPrint(
          '[HISTORY DEBUG] idx=$i id=${item['history_id']} bean=${item['bean_type_name'] ?? item['bean_type']} '
          'defect%=${item['defective_percent'] ?? shelfLifeDebug['defect_percentage']} '
          'shelfLifeMonths=${shelfLifeDebug['estimated_months'] ?? shelfLifeDebug['predicted_months'] ?? item['predicted_months']} '
          'shelfLifeDays=${shelfLifeDebug['predicted_days'] ?? item['predicted_days']} raw_keys=${item.keys}',
        );
      } catch (e) {
        debugPrint('[HISTORY DEBUG] idx=$i failed to log item: $e');
      }
    }

    if (!mounted) return;
    setState(() {
      _items = combined;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDark = theme.brightness == Brightness.dark;
    final Color pageBackground = isDark ? _darkBackground : _lightBackground;
    final Color cardBackground = isDark ? _darkCard : _lightCard;
    final Color borderColor = isDark ? _darkBorder : _lightBorder;
    final Color headerTextColor = isDark ? Colors.white : _lightHeaderText;
    final Color bodyTextColor = isDark
        ? Colors.white.withValues(alpha: 0.85)
        : _lightBodyText;
    return ColoredBox(
      color: pageBackground,
      child: Column(
        children: [
          _buildHeader(headerTextColor, pageBackground),
          _buildContent(
            context,
            colorScheme,
            pageBackground,
            cardBackground,
            borderColor,
            headerTextColor,
            bodyTextColor,
            isDark,
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(Color titleColor, Color background) {
    return Container(
      width: double.infinity,
      color: background,
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(
            vertical: AppConstants.headerPadding,
            horizontal: AppConstants.largePadding,
          ),
          child: Row(
            children: [
              const SizedBox(width: 36),
              Expanded(
                child: Text(
                  "History",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: titleColor,
                  ),
                ),
              ),
              const SizedBox(width: 36),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildContent(
    BuildContext context,
    ColorScheme colorScheme,
    Color pageBackground,
    Color cardBackground,
    Color borderColor,
    Color headerTextColor,
    Color bodyTextColor,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    return Expanded(
      child: Container(
        color: pageBackground,
        child: _loading
            ? Center(
                child: CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(
                    colorScheme.primary,
                  ),
                ),
              )
            : RefreshIndicator(
                onRefresh: () => _loadHistory(showLoadingIndicator: true),
                child: _items.isEmpty
                    ? LayoutBuilder(
                        builder: (context, constraints) {
                          return ListView(
                            physics: const AlwaysScrollableScrollPhysics(),
                            padding: const EdgeInsets.symmetric(
                              horizontal: AppConstants.largePadding,
                            ),
                            children: [
                              SizedBox(
                                height: constraints.maxHeight > 0
                                    ? constraints.maxHeight
                                    : 200,
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Text(
                                      "You don't have any history",
                                      style: theme.textTheme.titleSmall
                                          ?.copyWith(
                                            fontSize: 16,
                                            fontWeight: FontWeight.w600,
                                            color: AppColors.primaryBrown,
                                          ),
                                    ),
                                    const SizedBox(
                                      height: AppConstants.smallSpacing,
                                    ),
                                    Text(
                                      "Once you scan a coffee bean, results will show up here.",
                                      style: theme.textTheme.bodySmall
                                          ?.copyWith(
                                            fontSize: 12,
                                            color: AppColors.textGrey,
                                          ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          );
                        },
                      )
                    : ListView.separated(
                        physics: const AlwaysScrollableScrollPhysics(),
                        padding: const EdgeInsets.symmetric(
                          horizontal: AppConstants.largePadding,
                          vertical: AppConstants.largePadding,
                        ),
                        itemBuilder: (_, index) => _historyTile(
                          context,
                          colorScheme,
                          _items[index],
                          cardBackground,
                          borderColor,
                          headerTextColor,
                          bodyTextColor,
                          isDark,
                        ),
                        separatorBuilder: (_, __) =>
                            const SizedBox(height: AppConstants.mediumSpacing),
                        itemCount: _items.length,
                      ),
              ),
      ),
    );
  }

  Widget _historyTile(
    BuildContext context,
    ColorScheme colorScheme,
    dynamic item,
    Color cardBackground,
    Color borderColor,
    Color headerTextColor,
    Color bodyTextColor,
    bool isDark,
  ) {
    final String beanType = (item['bean_type_name'] ?? item['bean_type'] ?? 'Unknown').toString();
    final double defectivePct = _deriveDefectPercentage(item);
    final double healthyPct = _deriveHealthyPercentage(item, defectivePct);
    final Color effectiveBorder =
        isDark ? borderColor.withValues(alpha: 0.55) : borderColor;
    final Color effectiveCardColor =
        cardBackground.withValues(alpha: isDark ? 0.75 : 0.85);
    final Color iconBackground =
        (isDark ? _darkIconBackground : _lightIconBackground)
            .withValues(alpha: isDark ? 0.75 : 0.9);
    final String dateLabel = _formatHistoryTimestamp(item['created_at']);
    final String shelfLifeLabel = _formatShelfLifeLabel(item);
    return InkWell(
      onTap: () async {
        try {
          final historyId = item['history_id'];
          final localData = item['local_data'] as Map<String, dynamic>?;

          if (historyId == null && localData != null) {
            final predictionMap =
                localData['prediction'] as Map<String, dynamic>? ?? {};
            final probabilities = <String, double>{};
            final rawProbs = predictionMap['all_probabilities'];
            if (rawProbs is Map) {
              for (final entry in rawProbs.entries) {
                final value = entry.value;
                if (value is num) {
                  probabilities[entry.key.toString()] = value.toDouble();
                }
              }
            }
            final prediction = BeanPrediction(
              prediction: (predictionMap['prediction'] ?? beanType).toString(),
              confidence:
                  (predictionMap['confidence'] as num?)?.toDouble() ??
                  (item['confidence_score'] as num?)?.toDouble() ??
                  0.0,
              allProbabilities: probabilities,
            );

            final fallbackDefectivePct = _asDouble(item['defective_percent']);
            Map<String, dynamic>? localShelfLife = _cloneMap(localData['shelf_life']);
            if (fallbackDefectivePct != null && localShelfLife != null) {
              localShelfLife.putIfAbsent(
                'defect_percentage',
                () => fallbackDefectivePct,
              );
              localShelfLife.putIfAbsent(
                'defective_percent',
                () => fallbackDefectivePct,
              );
            }
            Map<String, dynamic>? localDefectDetection = _cloneMap(
              localData['defect_detection'],
            );
            if (localDefectDetection != null) {
              final summary =
                  _cloneMap(localDefectDetection['summary']) ??
                  <String, dynamic>{};
              if (fallbackDefectivePct != null) {
                summary.putIfAbsent(
                  'defect_percentage',
                  () => fallbackDefectivePct,
                );
                summary.putIfAbsent(
                  'defective_percent',
                  () => fallbackDefectivePct,
                );
              }
          _ensureSeverityConsistency(
            summary: summary,
            shelfLifeData: localShelfLife,
            fallbackPercentage: fallbackDefectivePct,
          );
              // Normalize months from days for local shelf life (override stale months)
              if (localShelfLife != null) {
                final double? pd = _asDouble(localShelfLife['predicted_days']) ??
                    _asDouble(localShelfLife['raw_prediction']);
                if (pd != null && pd > 0) {
                  final double pm = pd / 30.0;
                  localShelfLife['predicted_days'] = pd;
                  localShelfLife['predicted_months'] = pm;
                  localShelfLife['estimated_months'] = pm;
                }
              }
              localDefectDetection['summary'] = summary;
              
              // Ensure image_dimensions is preserved for proper defect annotation scaling
              final dynamic existingDims = localDefectDetection['image_dimensions'] ?? localDefectDetection['image_size'];
              if (existingDims is Map) {
                final dimsMap = Map<String, dynamic>.from(existingDims);
                final double? existingWidth = _asDouble(dimsMap['width']);
                final double? existingHeight = _asDouble(dimsMap['height']);
                if (existingWidth != null && existingHeight != null && existingWidth > 0 && existingHeight > 0) {
                  // Already has valid dimensions, keep them
                  localDefectDetection['image_dimensions'] = {
                    'width': existingWidth,
                    'height': existingHeight,
                  };
                }
              }
            } else {
              _ensureSeverityConsistency(
                summary: null,
                shelfLifeData: localShelfLife,
                fallbackPercentage: fallbackDefectivePct,
              );
              if (localShelfLife != null) {
                final double? pd = _asDouble(localShelfLife['predicted_days']) ??
                    _asDouble(localShelfLife['raw_prediction']);
                if (pd != null && pd > 0) {
                  final double pm = pd / 30.0;
                  localShelfLife['predicted_months'] ??= pm;
                  localShelfLife['estimated_months'] ??= pm;
                }
              }
            }

            if (!context.mounted) {
              return;
            }

            // Normalize months from days for local shelf life just before navigation
            if (localShelfLife != null) {
              final double? pdFinal = _asDouble(localShelfLife['predicted_days']) ??
                  _asDouble(localShelfLife['raw_prediction']);
              if (pdFinal != null && pdFinal > 0) {
                final double pmFinal = pdFinal / 30.0;
                localShelfLife['predicted_days'] = pdFinal;
                localShelfLife['predicted_months'] = pmFinal;
                localShelfLife['estimated_months'] = pmFinal;
              }
            }

            final navigator = Navigator.of(context);
            await navigator.push(
              MaterialPageRoute(
                builder: (_) => ResultsPage.history(
                  prediction: prediction,
                  imagePath:
                      (localData['image_path'] ?? item['image_url'] ?? '')
                          as String,
                  defectDetection: localDefectDetection,
                  shelfLife: localShelfLife,
                ),
              ),
            );
            return;
          }

        if (historyId == null) {
          return;
        }

        // Fetch full details then navigate to ResultsPage-like detail
        final details = await ApiService.fetchHistoryDetails(historyId);
        if (!mounted || details['success'] != true) return;
        final data = details['data'];

        // Build prediction from details to match live scan shape
        final prediction = BeanPrediction(
          prediction: data['bean_type']?['type_name'] ?? beanType,
          confidence:
              (data['shelf_life']?['confidence_score'] ??
                      item['confidence_score'] ??
                      healthyPct / 100)
                  .toDouble(),
          allProbabilities: const {},
        );

        final historyData = _cloneMap(data['history']);
        Map<String, dynamic>? shelfLifeData =
            _cloneMap(data['shelf_life']) ??
            _cloneMap(data['shelf_life_prediction']);
        Map<String, dynamic>? defectDetection =
            _cloneMap(data['defect_detection']);

        final double? historyDefectivePct =
            _asDouble(historyData?['defective_percent']);
        final double listDefectivePct = _deriveDefectPercentage(item);
        final double effectiveDefectivePct =
            historyDefectivePct ?? listDefectivePct;
        final Map<String, dynamic> shelfLifeMap =
            shelfLifeData ??= <String, dynamic>{};
        shelfLifeMap.putIfAbsent(
          'defect_percentage',
          () => effectiveDefectivePct,
        );
        shelfLifeMap.putIfAbsent(
          'defective_percent',
          () => effectiveDefectivePct,
        );
        // Normalize months from days (always trust days for detail view, override stale months)
        final double? pdServer = _asDouble(shelfLifeMap['predicted_days']) ??
            _asDouble(shelfLifeMap['raw_prediction']);
        if (pdServer != null && pdServer > 0) {
          final double pm = pdServer / 30.0;
          shelfLifeMap['predicted_days'] = pdServer;
          shelfLifeMap['estimated_months'] = pm;
          shelfLifeMap['predicted_months'] = pm;
        }

        if (defectDetection != null) {
          final summary =
              _cloneMap(defectDetection['summary']) ?? <String, dynamic>{};
          double? summaryPct = _asDouble(summary['defect_percentage']);
          if (summaryPct == null ||
              (summaryPct == 0 && effectiveDefectivePct > 0)) {
            summary['defect_percentage'] = effectiveDefectivePct;
            summaryPct = effectiveDefectivePct;
          }

          final dynamic severitySourceRaw =
              shelfLifeMap['severity'] ??
              shelfLifeMap['category'] ??
              historyData?['severity'];
          final String? incomingSeverity = severitySourceRaw is String
              ? severitySourceRaw.trim()
              : null;
          final String? existingSeverity =
              (summary['severity'] as String?)?.trim();
          if ((existingSeverity == null || existingSeverity.isEmpty) &&
              incomingSeverity != null &&
              incomingSeverity.isNotEmpty) {
            summary['severity'] = incomingSeverity;
          }

          _ensureSeverityConsistency(
            summary: summary,
            shelfLifeData: shelfLifeData,
            fallbackPercentage: summaryPct,
          );
          defectDetection['summary'] = summary;
          
          // Ensure image_dimensions is preserved for proper defect annotation scaling
          // First check if it's already in defectDetection
          final dynamic existingDims = defectDetection['image_dimensions'] ?? defectDetection['image_size'];
          if (existingDims is Map) {
            final dimsMap = Map<String, dynamic>.from(existingDims);
            final double? existingWidth = _asDouble(dimsMap['width']);
            final double? existingHeight = _asDouble(dimsMap['height']);
            if (existingWidth != null && existingHeight != null && existingWidth > 0 && existingHeight > 0) {
              // Already has valid dimensions, keep them
              defectDetection['image_dimensions'] = {
                'width': existingWidth,
                'height': existingHeight,
              };
            }
          } else {
            // Try to get from image data
            try {
              final dynamic imageData = data?['image'];
              if (imageData is Map) {
                final imageMap = Map<String, dynamic>.from(imageData);
                final double? imageWidth = _asDouble(imageMap['width'] ?? imageMap['image_width']);
                final double? imageHeight = _asDouble(imageMap['height'] ?? imageMap['image_height']);
                if (imageWidth != null && imageHeight != null && imageWidth > 0 && imageHeight > 0) {
                  defectDetection['image_dimensions'] = {
                    'width': imageWidth,
                    'height': imageHeight,
                  };
                }
              }
            } catch (e) {
              // If image data is not available, continue without image_dimensions
              debugPrint('Could not extract image dimensions from history data: $e');
            }
          }
        } else {
          _ensureSeverityConsistency(
            summary: null,
            shelfLifeData: shelfLifeData,
            fallbackPercentage: effectiveDefectivePct,
          );
        }

        if (!mounted || !context.mounted) {
          return;
        }
        final navigator = Navigator.of(context);
        await navigator.push(
          MaterialPageRoute(
            builder: (_) => ResultsPage.history(
              prediction: prediction,
              imagePath: (data['image']?['image_url'] ?? '') as String,
              defectDetection: defectDetection,
              shelfLife: shelfLifeData,
            ),
          ),
        );
        } catch (e, stackTrace) {
          debugPrint('Error opening history item: $e');
          debugPrint('Stack trace: $stackTrace');
          if (mounted && context.mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Error opening history: ${e.toString()}'),
                backgroundColor: Colors.red,
              ),
            );
          }
        }
      },
      child: Container(
        decoration: BoxDecoration(
          color: effectiveCardColor,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: effectiveBorder,
            width: AppConstants.thinBorder,
          ),
        ),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _HistoryCardIcon(
              background: iconBackground,
              borderColor: effectiveBorder,
            ),
            const SizedBox(width: AppConstants.mediumSpacing),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    dateLabel,
                    style: TextStyle(
                      color: bodyTextColor,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    beanType,
                    style: TextStyle(
                      color: headerTextColor.withValues(
                        alpha: isDark ? 0.95 : 1.0,
                      ),
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 4),
                ],
              ),
            ),
            Icon(
              Icons.chevron_right,
              color: headerTextColor.withValues(alpha: isDark ? 0.8 : 0.6),
            ),
          ],
        ),
      ),
    );
  }

  Map<String, dynamic>? _cloneMap(dynamic value) {
    if (value is Map) {
      return Map<String, dynamic>.from(value);
    }
    return null;
  }

  double? _asDouble(dynamic value) {
    if (value is num) return value.toDouble();
    if (value is String) {
      final normalized = value.trim();
      if (normalized.isEmpty) return null;
      final cleaned = normalized.replaceAll(RegExp(r'[^0-9\.\-]'), '');
      if (cleaned.isEmpty) return null;
      return double.tryParse(cleaned);
    }
    return null;
  }

  void _ensureSeverityConsistency({
    Map<String, dynamic>? summary,
    Map<String, dynamic>? shelfLifeData,
    double? fallbackPercentage,
  }) {
    if (summary == null && shelfLifeData == null) {
      return;
    }

    String? candidate = (summary?['severity'] as String?)?.trim();
    final dynamic shelfSeveritySource =
        shelfLifeData?['severity'] ?? shelfLifeData?['category'];
    final String? shelfSeverity = shelfSeveritySource is String
        ? shelfSeveritySource.trim()
        : null;

    if (candidate == null || candidate.isEmpty) {
      candidate = shelfSeverity;
    }

    final double? summaryPct = _asDouble(summary?['defect_percentage']);
    final double? shelfPct = _asDouble(
      shelfLifeData?['defect_percentage'] ??
          shelfLifeData?['defective_percent'],
    );
    final double? effectivePercentage =
        summaryPct ?? shelfPct ?? fallbackPercentage;

    final String? normalized = _normalizeSeverity(
      candidate,
      fallbackPercentage: effectivePercentage,
    );

    if (normalized != null) {
      summary?['severity'] = normalized;
      if (shelfLifeData != null) {
        shelfLifeData['severity'] = normalized;
      }
    }
  }

  String? _normalizeSeverity(String? severity, {double? fallbackPercentage}) {
    final trimmed = severity?.trim();
    String? normalized;
    if (trimmed != null && trimmed.isNotEmpty) {
      final lower = trimmed.toLowerCase();
      switch (lower) {
        case 'normal':
          normalized = 'normal';
          break;
        case 'mild':
        case 'moderate':
        case 'severe':
          normalized = lower;
          break;
        case 'excellent':
        case 'good':
        case 'optimal':
        case 'great':
        case 'low':
          normalized = 'mild';
          break;
        case 'warning':
        case 'fair':
        case 'medium':
          normalized = 'moderate';
          break;
        case 'critical':
        case 'expired':
        case 'poor':
        case 'high':
          normalized = 'severe';
          break;
        default:
          normalized = lower;
          break;
      }
    }

    final String? fallbackNormalized =
        fallbackPercentage != null ? _severityFromPercentage(fallbackPercentage) : null;

    if (normalized == null || normalized.isEmpty) {
      return fallbackNormalized;
    }

    if (fallbackNormalized != null &&
        _severityRank(fallbackNormalized) > _severityRank(normalized)) {
      return fallbackNormalized;
    }

    return normalized;
  }

  String _formatHistoryTimestamp(dynamic value) {
    if (value is! String || value.isEmpty) {
      return 'Date unavailable';
    }
    final parsed = DateTime.tryParse(value);
    if (parsed == null) return value;
    final local = parsed.toLocal();
    final mm = local.month.toString().padLeft(2, '0');
    final dd = local.day.toString().padLeft(2, '0');
    final yyyy = local.year.toString();
    final hh = local.hour.toString().padLeft(2, '0');
    final min = local.minute.toString().padLeft(2, '0');
    return '$mm/$dd/$yyyy at $hh:$min';
  }

  String _formatShelfLifeLabel(dynamic item) {
    final double? months = _extractShelfLifeMonths(item);
    if (months != null && months > 0) {
      final String display = months >= 9
          ? months.toStringAsFixed(0)
          : months.toStringAsFixed(1);
      return 'Estimated shelf life: $display months';
    }
    return 'Estimated shelf life: unavailable';
  }

  double? _extractShelfLifeMonths(dynamic item) {
    if (item is! Map<String, dynamic>) return null;
    final bool localOnly = item['history_id'] == null;

    final Map<String, dynamic>? shelfLifeMap =
        _deriveShelfLifeMap(Map<String, dynamic>.from(item));

    // Prefer enriched shelf life attached to history rows (backend now sends this)
    if (shelfLifeMap != null) {
      final double? enrichedDays =
          _asDouble(shelfLifeMap['predicted_days']) ?? _asDouble(shelfLifeMap['raw_prediction']);
      if (enrichedDays != null && enrichedDays > 0) {
        return double.parse((enrichedDays / 30.0).toStringAsFixed(1));
      }
      final double? enrichedMonths =
          _asDouble(shelfLifeMap['estimated_months']) ?? _asDouble(shelfLifeMap['predicted_months']);
      if (enrichedMonths != null && enrichedMonths > 0 && !enrichedMonths.isNaN) {
        return enrichedMonths;
      }
    }

    // For local entries that still carry only the stale 180-day cache, recompute immediately.
    if (localOnly) {
      return _recomputeLocalShelfLifeMonths(item);
    }

    final double? itemDays = _asDouble(item['predicted_days']);
    if (itemDays != null && itemDays > 0) {
      return double.parse((itemDays / 30.0).toStringAsFixed(1));
    }

    final double? itemMonths =
        _asDouble(item['predicted_months']) ?? _asDouble(item['estimated_months']);
    if (itemMonths != null && itemMonths > 0 && !itemMonths.isNaN) {
      return itemMonths;
    }

    final double? storedDays =
        _asDouble(shelfLifeMap?['predicted_days']) ??
        _asDouble(shelfLifeMap?['raw_prediction']);
    if (storedDays != null && storedDays > 0) {
      return double.parse((storedDays / 30.0).toStringAsFixed(1));
    }

    final double? storedMonths =
        _asDouble(shelfLifeMap?['predicted_months']) ??
        _asDouble(shelfLifeMap?['estimated_months']);
    if (storedMonths != null && storedMonths > 0 && !storedMonths.isNaN) {
        return storedMonths;
    }

    // Fallback: recompute using weighted defect score (includes good beans).
    final Map<String, dynamic>? defectDetection = _resolveDefectDetection(Map<String, dynamic>.from(item));
    final List<dynamic> detections = _extractDetections(defectDetection);
    final Map<String, dynamic>? shelfCounts = shelfLifeMap?['defect_counts'] is Map
        ? Map<String, dynamic>.from(shelfLifeMap!['defect_counts'] as Map)
        : null;
    final double weightedDays = _historyWeightedShelfLifeDays(
          typeCounts: shelfCounts,
          detections: detections,
        ) ??
        30.0;

    return double.parse((weightedDays / 30.0).toStringAsFixed(1));
  }

  double? _preferDaysWhenMismatch(double? days, double? months) {
    // Deprecated: no longer used
    return months;
  }

  double _deriveDefectPercentage(dynamic raw) {
    if (raw is! Map<String, dynamic>) return 0.0;
    final Map<String, dynamic> map = raw;

    // Prefer direct values provided by the backend row first
    final double? directPct =
        _asDouble(map['defective_percent']) ?? _asDouble(map['defect_percentage']);
    if (directPct != null && directPct >= 0) {
      return directPct;
    }

    final Map<String, dynamic>? shelfLife = _deriveShelfLifeMap(map);
    // Prefer enriched shelf life defect percentage if present
    final double? enrichedPct =
        _asDouble(shelfLife?['defect_percentage']) ?? _asDouble(map['defect_percentage']);
    if (enrichedPct != null && enrichedPct > 0) {
      return enrichedPct;
    }

    final Map<String, dynamic>? defectDetection = _resolveDefectDetection(map);
    final List<dynamic> detections = _extractDetections(defectDetection);
    final Map<String, dynamic> summary = _buildHistoryDefectSummary(
      rawMap: map,
      defectDetection: defectDetection,
      shelfLife: shelfLife,
      detections: detections,
    );

    final double? shelfLifePct = _asDouble(
      shelfLife?['defect_percentage'] ?? shelfLife?['defective_percent'],
    );
    final double? summaryPct = _asDouble(summary['defect_percentage']);

    double? resolvedCandidate = summaryPct;
    if (resolvedCandidate == null || resolvedCandidate <= 0) {
      resolvedCandidate = shelfLifePct;
    }
    if (resolvedCandidate == null || resolvedCandidate <= 0) {
      resolvedCandidate = _asDouble(map['defective_percent']) ??
          _asDouble(_cloneMap(map['history'])?['defective_percent']) ??
          _asDouble(_cloneMap(map['local_data'])?['defective_percent']) ??
          _asDouble(_cloneMap(map['localData'])?['defective_percent']);
    }

    final int totalDefects =
        summary['total_defects'] is num
            ? (summary['total_defects'] as num).toInt()
            : detections.length;

    double resolvedPctValue;
    if (resolvedCandidate != null && resolvedCandidate > 0) {
      resolvedPctValue = resolvedCandidate;
    } else {
      if (totalDefects > 0) {
        double detectionDrivenPct =
            _estimateDefectPercentageFromDetections(detections) ??
                (totalDefects / math.max(totalDefects, 12)) * 100.0;
        resolvedPctValue = detectionDrivenPct;
      } else {
        resolvedPctValue = 0.0;
      }
    }

    return resolvedPctValue.clamp(0.0, 100.0);
  }

  double _deriveHealthyPercentage(dynamic raw, double fallbackDefective) {
    if (raw is! Map<String, dynamic>) {
      final num clamped = (100.0 - fallbackDefective).clamp(0.0, 100.0);
      return clamped.toDouble();
    }
    final Map<String, dynamic> map = raw;
    double? value = _asDouble(map['healthy_percent']);
    value ??= _asDouble(_cloneMap(map['history'])?['healthy_percent']);
    value ??= _asDouble(_cloneMap(map['health_score'])?['percentage']);
    if (value == null || value.isNaN) {
      final num clamped = (100.0 - fallbackDefective).clamp(0.0, 100.0);
      return clamped.toDouble();
    }
    final num clampedValue = value.clamp(0.0, 100.0);
    return clampedValue.toDouble();
  }

  Map<String, dynamic>? _deriveShelfLifeMap(Map<String, dynamic> map) {
    Map<String, dynamic>? resolveShelfLife(Map<String, dynamic>? source) {
      if (source == null) return null;
      return _cloneMap(source['shelf_life']) ??
          _cloneMap(source['shelf_life_prediction']) ??
          _cloneMap(source['shelfLife']) ??
          _cloneMap(source['shelfLifePrediction']) ??
          _cloneMap(source['shelfLifeData']);
    }

    return resolveShelfLife(map) ??
        resolveShelfLife(_cloneMap(map['history'])) ??
        resolveShelfLife(_cloneMap(map['local_data'])) ??
        resolveShelfLife(_cloneMap(map['localData'])) ??
        resolveShelfLife(_cloneMap(map['payload'])) ??
        resolveShelfLife(_cloneMap(map['data']));
  }

  static const Map<String, int> _baseShelfLifeDaysHistory = {
    'arabica': 900, // 30 months baseline
    'liberica': 840, // 28 months baseline
    'excelsa': 780, // 26 months baseline
    'robusta': 750, // 25 months baseline
    'other': 600, // 20 months baseline
  };

  static const Map<String, double> _historyDefectWeights = {
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

  static const double _historyBaselineShelfLifeDays = 900.0; // ~30 months midpoint for clean beans
  static const Map<String, double> _historyDefectReductions = {
    'fully_black': 0.75,
    'roasted': 0.833, // ~5 months target
    'insect': 0.73,
    'broken': 0.25,
    'good_beans': 0.0,
  };

  String? _historyShelfLifeBucket(String? raw) {
    if (raw == null) return null;
    final norm = raw.toLowerCase().replaceAll(RegExp(r'[_-]+'), ' ').trim();
    if (norm.contains('fully black') || norm.contains('full black') || norm.contains('black bean')) return 'fully_black';
    if (norm.contains('roast')) return 'roasted';
    if (norm.contains('insect')) return 'insect';
    if (norm.contains('broken') || norm.contains('cut')) return 'broken';
    if (norm.contains('good bean')) return 'good_beans';
    return null;
  }

  double? _historyWeightedShelfLifeDays({
    Map<String, dynamic>? typeCounts,
    List<dynamic>? detections,
  }) {
    final Map<String, int> counts = {};

    void bump(String? raw, int amount) {
      final bucket = _historyShelfLifeBucket(raw);
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
        final String? rawType = (detection['defect_type'] ?? detection['label'] ?? detection['class'] ?? detection['type'])
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
      final weight = _historyDefectWeights[key] ?? 0.0;
      weightedSum += weight * value;
    });

    double reductionSum = 0;
    counts.forEach((key, value) {
      final reduction = _historyDefectReductions[key] ?? 0.3;
      reductionSum += reduction * value;
    });

    final double avgReduction = (reductionSum / totalBeans).clamp(0.0, 0.9);
    final double days =
        (_historyBaselineShelfLifeDays * (1.0 - avgReduction)).clamp(0.0, _historyBaselineShelfLifeDays);
    return days;
  }

  bool _historyIsGoodBean(String? defectType) {
    if (defectType == null) return false;
    final norm = defectType.toLowerCase().replaceAll(RegExp(r'[_-]+'), ' ').trim();
    return norm.contains('good bean');
  }

  String _historyNormalizeDefectKey(String raw) {
    return raw.toLowerCase().replaceAll(RegExp(r'[_-]+'), ' ').trim();
  }

  double _historyWeightedScore(List<dynamic> detections) {
    double sum = 0;
    int count = 0;
    for (final d in detections) {
      if (d is! Map) continue;
      final String? t = (d['defect_type'] ?? d['label'] ?? d['class'] ?? d['type'])?.toString();
      if (t == null) continue;
      final norm = _historyNormalizeDefectKey(t);
      if (_historyIsGoodBean(norm)) continue;
      final double w =
          _historyDefectWeights[norm] ?? _historyDefectWeights[norm.replaceAll(' ', '_')] ?? 0.3;
      sum += w;
      count += 1;
    }
    if (count == 0) return 0.0;
    return (sum / count).clamp(0.0, 1.0);
  }

  double _historyBaseDays(String? beanType) {
    final lower = beanType?.toLowerCase() ?? '';
    if (lower.contains('arabica')) return _baseShelfLifeDaysHistory['arabica']!.toDouble();
    if (lower.contains('liberica')) return _baseShelfLifeDaysHistory['liberica']!.toDouble();
    if (lower.contains('excelsa')) return _baseShelfLifeDaysHistory['excelsa']!.toDouble();
    if (lower.contains('robusta')) return _baseShelfLifeDaysHistory['robusta']!.toDouble();
    return _baseShelfLifeDaysHistory['other']!.toDouble();
  }

  double? _historyOverrideDaysForRank(int rank) {
    switch (rank) {
      case 4:
        return 0.0; // Fully black: discard immediately
      case 3:
        return 135.0; // Insect damage: ~3–6 months
      case 2:
        return 547.0; // Broken/Cut: ~1–2 years
      case 1:
        return 10.0; // Roasted/heat damaged: ~1–2 weeks
      default:
        return null;
    }
  }

  double _historyShelfMultiplierFromDefectPct(double defectPct) {
    final double score = (defectPct / 100.0).clamp(0.0, 1.0);
    final double linear = 1.0 - (0.7 * score);
    return linear.clamp(0.25, 1.0);
  }

  double? _recomputeLocalShelfLifeMonths(Map<String, dynamic> item) {
    final Map<String, dynamic>? defectDetection = _resolveDefectDetection(Map<String, dynamic>.from(item));
    final List<dynamic> detections = _extractDetections(defectDetection);
    final Map<String, dynamic>? shelfCounts = item['defect_counts'] is Map
        ? Map<String, dynamic>.from(item['defect_counts'] as Map)
        : null;

    final double weightedDays = _historyWeightedShelfLifeDays(
          typeCounts: shelfCounts,
          detections: detections,
        ) ??
        30.0;

    return weightedDays / 30.0;
  }


  double _historyShelfMultiplierFromScore(double score) {
    final double linear = 1.0 - (0.7 * score);
    return linear.clamp(0.25, 1.0);
  }

  int _historyDefectRank(String? defectType) {
    if (defectType == null) return 0;
    final norm = defectType.toLowerCase().replaceAll(RegExp(r'[_-]+'), ' ').trim();
    if (norm.contains('fully black') || norm.contains('full black') || norm.contains('black bean')) return 4;
    if (norm.contains('insect')) return 3;
    if (norm.contains('broken') || norm.contains('cut')) return 2;
    if (norm.contains('roast')) return 1;
    return 0;
  }

  int _historyHighestRank(List<dynamic> detections) {
    int rank = 0;
    for (final d in detections) {
      if (d is! Map) continue;
      final String? t = (d['defect_type'] ?? d['label'] ?? d['class'] ?? d['type'])?.toString();
      rank = math.max(rank, _historyDefectRank(t));
    }
    return rank;
  }

  double? _historyRankOverrideDays(int rank) {
    switch (rank) {
      case 4:
        return 360.0; // Fully black beans: 10–14 months (midpoint ≈12 months)
      case 3:
        return 420.0; // Insect damage: 12–16 months (midpoint ≈14 months)
      case 2:
        return 675.0; // Broken / Cut beans: 18–27 months (midpoint ≈22.5 months)
      case 1:
        return 10.0; // 1–2 weeks midpoint
      default:
        return null;
    }
  }

  Map<String, dynamic>? _resolveDefectDetection(Map<String, dynamic> map) {
    Map<String, dynamic>? resolve(Map<String, dynamic>? source) {
      if (source == null) return null;
      return _cloneMap(source['defect_detection']) ??
          _cloneMap(source['defectDetection']) ??
          _cloneMap(source['defect']) ??
          _cloneMap(source['detection']);
    }

    return resolve(map) ??
        resolve(_cloneMap(map['history'])) ??
        resolve(_cloneMap(map['local_data'])) ??
        resolve(_cloneMap(map['localData'])) ??
        resolve(_cloneMap(map['payload'])) ??
        resolve(_cloneMap(map['data']));
  }

  Map<String, dynamic> _buildHistoryDefectSummary({
    required Map<String, dynamic> rawMap,
    required Map<String, dynamic>? defectDetection,
    required Map<String, dynamic>? shelfLife,
    required List<dynamic> detections,
  }) {
    Map<String, dynamic> summary = <String, dynamic>{};

    if (defectDetection?['summary'] is Map<String, dynamic>) {
      summary =
          Map<String, dynamic>.from(defectDetection!['summary'] as Map);
    } else if (defectDetection != null) {
      final Map<String, dynamic> dd =
          Map<String, dynamic>.from(defectDetection);
      summary = {
        'total_defects':
            (dd['total_defects'] as num?)?.toInt() ?? detections.length,
        'defect_percentage': _asDouble(dd['defect_percentage']),
        'quality_grade': dd['quality_grade'],
        'severity': dd['severity'],
        'confidence': _asDouble(dd['confidence']),
      };
    }

    final double? shelfPct = _asDouble(
      shelfLife?['defect_percentage'] ?? shelfLife?['defective_percent'],
    );
    final int summaryCount =
        (summary['total_defects'] as num?)?.toInt() ?? 0;
    summary['total_defects'] =
        summaryCount > 0 ? math.max(summaryCount, detections.length) : detections.length;

    final double? summaryPct = _asDouble(summary['defect_percentage']);
    if ((summaryPct == null || summaryPct <= 0) &&
        shelfPct != null &&
        shelfPct > 0) {
      summary['defect_percentage'] = shelfPct;
    }

    if (summary['defect_percentage'] == null &&
        summary['total_defects'] == 0 &&
        shelfPct != null) {
      summary['defect_percentage'] = shelfPct;
    }

    if (summary['defect_percentage'] == null) {
      summary['defect_percentage'] =
          _asDouble(shelfLife?['defective_percent']);
    }

    summary['defect_percentage'] ??=
        _asDouble(rawMap['defective_percent']) ??
            _asDouble(_cloneMap(rawMap['history'])?['defective_percent']) ??
            _asDouble(_cloneMap(rawMap['local_data'])?['defective_percent']) ??
            _asDouble(_cloneMap(rawMap['localData'])?['defective_percent']);

    return summary;
  }

  List<dynamic> _extractDetections(Map<String, dynamic>? defectDetection) {
    if (defectDetection == null) return const [];
    if (defectDetection['detections'] is List) {
      return List<dynamic>.from(defectDetection['detections'] as List);
    }
    final Map<String, dynamic> dd =
        Map<String, dynamic>.from(defectDetection);
    final Map<String, dynamic> coords = Map<String, dynamic>.from(
      (dd['defect_coordinates'] as Map?) ?? <String, dynamic>{},
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
      },
    ];
  }

  double? _estimateDefectPercentageFromDetections(
    List<dynamic> detections,
  ) {
    if (detections.isEmpty) {
      return null;
    }

    double totalConfidence = 0;
    for (final detection in detections) {
      final double confidence =
          (detection['confidence'] as num?)?.toDouble() ?? 0.5;
      totalConfidence += confidence.clamp(0.0, 1.0);
    }

    final double averageConfidence = totalConfidence / detections.length;
    final double normalizedConfidence = averageConfidence.clamp(0.3, 0.95);
    final double countFactor = math.min(1.0, detections.length / 3.0);
    final double base =
        32.0 + (detections.length * 18.0).clamp(0.0, 54.0);
    final double confidenceAdjustment = (normalizedConfidence - 0.5) * 50.0;
    final double countAdjustment = countFactor * 35.0;
    final double estimated =
        base + confidenceAdjustment + countAdjustment;
    return estimated.clamp(24.0, 95.0);
  }

  int _severityRank(String? severity) {
    switch (severity) {
      case 'normal':
        return 0;
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

  String _severityFromPercentage(double percentage) {
    if (percentage < 22) return 'mild';
    if (percentage < 78) return 'moderate';
    return 'severe';
  }
}

class _HistoryCardIcon extends StatelessWidget {
  final Color background;
  final Color borderColor;

  const _HistoryCardIcon({required this.background, required this.borderColor});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 52,
      height: 52,
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor, width: AppConstants.thinBorder),
      ),
      clipBehavior: Clip.antiAlias,
      child: Image.asset(
        'assets/icons/BeanScan Logo History Content.png',
        fit: BoxFit.cover,
      ),
    );
  }
}
