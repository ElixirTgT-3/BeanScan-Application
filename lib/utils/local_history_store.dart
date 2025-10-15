import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import 'api_service.dart' show BeanPrediction;

class CachedHistoryEntry {
  final int? historyId;
  final String beanType;
  final double healthyPercent;
  final double defectivePercent;
  final double confidenceScore;
  final String? imagePath;
  final DateTime createdAt;
  final Map<String, dynamic>? predictionData;
  final Map<String, dynamic>? defectDetection;
  final Map<String, dynamic>? shelfLife;

  CachedHistoryEntry({
    this.historyId,
    required this.beanType,
    required this.healthyPercent,
    required this.defectivePercent,
    required this.confidenceScore,
    this.imagePath,
    required this.createdAt,
    this.predictionData,
    this.defectDetection,
    this.shelfLife,
  });

  Map<String, dynamic> toApiLikeJson() {
    return {
      'history_id': historyId,
      'bean_type_name': beanType,
      'healthy_percent': healthyPercent,
      'defective_percent': defectivePercent,
      'confidence_score': confidenceScore,
      'image_url': imagePath,
      'created_at': createdAt.toIso8601String(),
      'source': 'local',
      if (predictionData != null || defectDetection != null || shelfLife != null)
        'local_data': {
          if (predictionData != null) 'prediction': predictionData,
          if (defectDetection != null) 'defect_detection': defectDetection,
          if (shelfLife != null) 'shelf_life': shelfLife,
          if (imagePath != null) 'image_path': imagePath,
        },
    };
  }

  Map<String, dynamic> toJson() {
    return {
      'history_id': historyId,
      'bean_type': beanType,
      'healthy_percent': healthyPercent,
      'defective_percent': defectivePercent,
      'confidence_score': confidenceScore,
      'image_path': imagePath,
      'created_at': createdAt.toIso8601String(),
      'prediction': predictionData,
      'defect_detection': defectDetection,
      'shelf_life': shelfLife,
    };
  }

  factory CachedHistoryEntry.fromJson(Map<String, dynamic> json) {
    return CachedHistoryEntry(
      historyId: json['history_id'] as int?,
      beanType: (json['bean_type'] ?? json['bean_type_name'] ?? 'Unknown').toString(),
      healthyPercent: (json['healthy_percent'] as num?)?.toDouble() ?? 0,
      defectivePercent: (json['defective_percent'] as num?)?.toDouble() ?? 0,
      confidenceScore: (json['confidence_score'] as num?)?.toDouble() ?? 0,
      imagePath: json['image_path'] as String?,
      createdAt: DateTime.tryParse(json['created_at'] as String? ?? '') ?? DateTime.now(),
      predictionData: json['prediction'] as Map<String, dynamic>?,
      defectDetection: json['defect_detection'] as Map<String, dynamic>?,
      shelfLife: json['shelf_life'] as Map<String, dynamic>?,
    );
  }

  static CachedHistoryEntry fromScanResponse({
    required Map<String, dynamic> response,
    required BeanPrediction prediction,
    required double healthyPercent,
    required double defectivePercent,
    required String? imagePath,
    Map<String, dynamic>? defectDetection,
    Map<String, dynamic>? shelfLife,
  }) {
    final predictionMap = {
      'prediction': prediction.prediction,
      'confidence': prediction.confidence,
      'all_probabilities': prediction.allProbabilities,
    };

    return CachedHistoryEntry(
      historyId: response['history_id'] as int?,
      beanType: prediction.prediction,
      healthyPercent: healthyPercent,
      defectivePercent: defectivePercent,
      confidenceScore: prediction.confidence,
      imagePath: imagePath,
      createdAt: DateTime.now(),
      predictionData: predictionMap,
      defectDetection: defectDetection,
      shelfLife: shelfLife,
    );
  }
}

class LocalHistoryStore {
  static const _prefsKey = 'local_history_entries';
  static const _maxEntries = 50;

  static Future<List<Map<String, dynamic>>> getEntries() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getStringList(_prefsKey) ?? [];
    final entries = <Map<String, dynamic>>[];
    for (final item in raw) {
      try {
        final decoded = json.decode(item) as Map<String, dynamic>;
        final entry = CachedHistoryEntry.fromJson(decoded);
        entries.add(entry.toApiLikeJson());
      } catch (_) {
        // ignore malformed entry
      }
    }
    return entries;
  }

  static Future<void> addEntry(CachedHistoryEntry entry) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getStringList(_prefsKey) ?? [];
    raw.insert(0, json.encode(entry.toJson()));
    if (raw.length > _maxEntries) {
      raw.removeRange(_maxEntries, raw.length);
    }
    await prefs.setStringList(_prefsKey, raw);
  }

  static Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_prefsKey);
  }
}
