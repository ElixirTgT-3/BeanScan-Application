import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'app_logger.dart';

void _logApiService(
  String message, {
  Object? error,
  StackTrace? stackTrace,
}) {
  logDebug(
    'ApiService',
    message,
    error: error,
    stackTrace: stackTrace,
  );
}

class BeanPrediction {
  final String prediction;
  final double confidence;
  final Map<String, double> allProbabilities;

  BeanPrediction({
    required this.prediction,
    required this.confidence,
    required this.allProbabilities,
  });

  factory BeanPrediction.fromJson(Map<String, dynamic> json) {
    final dynamic rawPrediction = json['prediction'] ??
        json['predicted_class'] ??
        json['class'] ??
        json['bean_type'] ??
        json['bean_type_name'];

    final double confidence = (() {
      final dynamic raw = json['confidence'] ??
          json['confidence_score'] ??
          json['probability'];
      return (raw is num) ? raw.toDouble() : 0.0;
    })();

    final Map<String, double> probabilityMap = <String, double>{};
    final dynamic rawProbabilities =
        json['all_probabilities'] ?? json['probabilities'];

    if (rawProbabilities is Map) {
      rawProbabilities.forEach((key, value) {
        if (value is num) {
          probabilityMap[key.toString()] = value.toDouble();
        }
      });
    } else if (rawProbabilities is List) {
      const List<String> defaultClasses = <String>[
        'Arabica',
        'Robusta',
        'Liberica',
        'Excelsa',
      ];
      final List<dynamic>? providedClasses = json['classes'] is List
          ? (json['classes'] as List).cast<dynamic>()
          : null;
      for (int i = 0; i < rawProbabilities.length; i++) {
        final dynamic value = rawProbabilities[i];
        if (value is! num) continue;
        final String label;
        if (providedClasses != null && i < providedClasses.length) {
          label = providedClasses[i]?.toString() ?? 'Class_$i';
        } else if (i < defaultClasses.length) {
          label = defaultClasses[i];
        } else {
          label = 'Class_$i';
        }
        probabilityMap[label] = value.toDouble();
      }
    }

    return BeanPrediction(
      prediction: rawPrediction?.toString() ?? '',
      confidence: confidence,
      allProbabilities: probabilityMap,
    );
  }
}

class DefectDetection {
  final bool success;
  final List<Defect> detections;
  final DefectSummary summary;
  final String? error;

  DefectDetection({
    required this.success,
    required this.detections,
    required this.summary,
    this.error,
  });

  factory DefectDetection.fromJson(Map<String, dynamic> json) {
    return DefectDetection(
      success: json['success'] ?? false,
      detections: (json['detections'] as List<dynamic>?)
          ?.map((d) => Defect.fromJson(d))
          .toList() ?? [],
      summary: DefectSummary.fromJson(json['summary'] ?? {}),
      error: json['error'],
    );
  }
}

class Defect {
  final List<double> bbox;
  final double confidence;
  final String defectType;
  final DefectCoordinates coordinates;
  final double area;
  final DefectCenter center;

  Defect({
    required this.bbox,
    required this.confidence,
    required this.defectType,
    required this.coordinates,
    required this.area,
    required this.center,
  });

  factory Defect.fromJson(Map<String, dynamic> json) {
    return Defect(
      bbox: List<double>.from(json['bbox'] ?? []),
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      defectType: json['defect_type'] ?? '',
      coordinates: DefectCoordinates.fromJson(json['coordinates'] ?? {}),
      area: (json['area'] ?? 0.0).toDouble(),
      center: DefectCenter.fromJson(json['center'] ?? {}),
    );
  }
}

class DefectCoordinates {
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  DefectCoordinates({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  factory DefectCoordinates.fromJson(Map<String, dynamic> json) {
    return DefectCoordinates(
      x1: (json['x1'] ?? 0.0).toDouble(),
      y1: (json['y1'] ?? 0.0).toDouble(),
      x2: (json['x2'] ?? 0.0).toDouble(),
      y2: (json['y2'] ?? 0.0).toDouble(),
    );
  }
}

class DefectCenter {
  final double x;
  final double y;

  DefectCenter({
    required this.x,
    required this.y,
  });

  factory DefectCenter.fromJson(Map<String, dynamic> json) {
    return DefectCenter(
      x: (json['x'] ?? 0.0).toDouble(),
      y: (json['y'] ?? 0.0).toDouble(),
    );
  }
}

class DefectSummary {
  final int totalDefects;
  final Map<String, int> defectTypes;
  final double defectPercentage;
  final double qualityScore;
  final String qualityGrade;

  DefectSummary({
    required this.totalDefects,
    required this.defectTypes,
    required this.defectPercentage,
    required this.qualityScore,
    required this.qualityGrade,
  });

  factory DefectSummary.fromJson(Map<String, dynamic> json) {
    return DefectSummary(
      totalDefects: json['total_defects'] ?? 0,
      defectTypes: Map<String, int>.from(json['defect_types'] ?? {}),
      defectPercentage: (json['defect_percentage'] ?? 0.0).toDouble(),
      qualityScore: (json['quality_score'] ?? 0.0).toDouble(),
      qualityGrade: json['quality_grade'] ?? 'Unknown',
    );
  }
}

class ApiService {
  // Allow overriding via --dart-define=API_BASE_URL=... and --dart-define=ANDROID_API_BASE_URL=...
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8000',
  );
  
  // For Android device/emulator, set with --dart-define=ANDROID_API_BASE_URL=http://<IP>:8000
  static const String androidBaseUrl = String.fromEnvironment(
    'ANDROID_API_BASE_URL',
    defaultValue: 'http://192.168.0.63:8000',
  );
  
  static String? _resolvedApiUrl;
  static String get apiUrl {
    if (_resolvedApiUrl != null) return _resolvedApiUrl!;
    final url = Platform.isAndroid ? androidBaseUrl : baseUrl;
    // Log the URL being used (only once)
    if (_resolvedApiUrl == null) {
      _logApiService('Using API URL: $url (Platform: ${Platform.isAndroid ? "Android" : "Other"})');
      _logApiService('baseUrl from env: $baseUrl');
      if (Platform.isAndroid) {
        _logApiService('androidBaseUrl from env: $androidBaseUrl');
      }
    }
    return url;
  }

  /// Check if the API is healthy
  static Future<bool> checkHealth() async {
    // Always prioritize the dart-define URL first
    final primaryUrl = apiUrl;
    _logApiService('Health check starting with primary URL: $primaryUrl');
    _logApiService('baseUrl: $baseUrl, androidBaseUrl: $androidBaseUrl');
    
    final candidates = <String>[
      primaryUrl, // Try dart-define URL first
      if (Platform.isAndroid) ...[
        // Android emulator default host mapping (only if not using dart-define)
        if (!primaryUrl.contains('192.168') && !primaryUrl.contains('10.0.2.2'))
          'http://10.0.2.2:8000',
        // Genymotion emulator
        if (!primaryUrl.contains('192.168') && !primaryUrl.contains('10.0.3.2'))
          'http://10.0.3.2:8000',
      ],
      // Common local fallbacks (only if not using dart-define)
      if (!primaryUrl.contains('localhost') && !primaryUrl.contains('127.0.0.1')) ...[
        'http://localhost:8000',
        'http://127.0.0.1:8000',
      ],
    ];

    for (final url in candidates) {
      try {
        _logApiService('Trying health check: $url');
        final response = await http.get(
          Uri.parse('$url/health'),
          headers: {'Content-Type': 'application/json'},
        ).timeout(const Duration(seconds: 5));
        if (response.statusCode == 200) {
          _resolvedApiUrl = url;
          _logApiService('✓ API reachable at: $url');
          if (url != primaryUrl) {
            _logApiService('⚠ Using fallback URL instead of dart-define URL');
          }
          return true;
        } else {
          _logApiService('Health check returned status ${response.statusCode} for $url');
        }
      } catch (e, stackTrace) {
        // Try next candidate
        _logApiService(
          '✗ Health check failed for $url',
          error: e,
          stackTrace: stackTrace,
        );
      }
    }
    _logApiService('✗ All health check candidates failed');
    return false;
  }

  /// Predict bean type from image file (legacy method - use scanBeanImage instead)
  static Future<BeanPrediction?> predictBeanType(File imageFile) async {
    try {
      // Use the scan endpoint which includes both classification and defect detection
      final result = await scanBeanImage(imageFile);
      if (result['success'] && result['data'] != null) {
        final predictionData = result['data']['prediction'];
        return BeanPrediction.fromJson(predictionData);
      }
      return null;
    } catch (e, stackTrace) {
      _logApiService(
        'Prediction failed',
        error: e,
        stackTrace: stackTrace,
      );
      return null;
    }
  }

  /// Get prediction with detailed error handling
  static Future<Map<String, dynamic>> predictBeanTypeWithErrorHandling(File imageFile) async {
    try {
      // First check if API is available
      final isHealthy = await checkHealth();
      if (!isHealthy) {
        return {
          'success': false,
          'error': 'API server is not available. Please make sure the backend is running.',
        };
      }

      // Make prediction
      final prediction = await predictBeanType(imageFile);
      if (prediction != null) {
        return {
          'success': true,
          'prediction': prediction,
        };
      } else {
        return {
          'success': false,
          'error': 'Failed to get prediction from API',
        };
      }
    } catch (e, stackTrace) {
      _logApiService(
        'predictBeanTypeWithErrorHandling failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  /// Scan bean image with both classification and defect detection
  static Future<Map<String, dynamic>> scanBeanImage(File imageFile) async {
    try {
      // First check if API is available
      final isHealthy = await checkHealth();
      if (!isHealthy) {
        return {
          'success': false,
          'error': 'API server is not available. Please make sure the backend is running.',
        };
      }

      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiUrl/api/v1/yolo/predict'),
      );

      // Add the image file
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
        ),
      );

      // Add optional device identifier
      final deviceId = await _getDeviceId();
      if (deviceId != null && deviceId.isNotEmpty) {
        request.fields['device_id'] = deviceId;
      }

      // Send the request
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonResponse = json.decode(response.body);
        final normalized = _normalizeYoloResponse(jsonResponse);
        return {'success': true, 'data': normalized};
      } else {
        _logApiService('API Error: ${response.statusCode} - ${response.body}');
        return {
          'success': false,
          'error': 'API Error: ${response.statusCode} - ${response.body}',
        };
      }
    } catch (e, stackTrace) {
      _logApiService(
        'Scan failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  static Map<String, dynamic> _normalizeYoloResponse(Map<String, dynamic> jsonResponse) {
    // Bean detections (classification)
    final List<dynamic> rawDetections = (jsonResponse['detections'] as List?) ?? const [];
    final Map<String, dynamic> classMap = {};
    final rawClasses = jsonResponse['classes'];
    if (rawClasses is Map) {
      rawClasses.forEach((key, value) => classMap[key.toString()] = value.toString());
    }

    final Map<String, dynamic> imageSize = jsonResponse['image_size'] is Map
        ? Map<String, dynamic>.from(jsonResponse['image_size'] as Map)
        : const <String, dynamic>{};

    final List<Map<String, dynamic>> beanDetections = [];
    for (final det in rawDetections) {
      if (det is! Map) continue;
      final detMap = Map<String, dynamic>.from(det);
      final bbox = detMap['bbox'];
      Map<String, dynamic> coords = {};
      if (bbox is Map) {
        coords = {
          'x1': (bbox['x1'] as num?)?.toDouble() ?? 0.0,
          'y1': (bbox['y1'] as num?)?.toDouble() ?? 0.0,
          'x2': (bbox['x2'] as num?)?.toDouble() ?? 0.0,
          'y2': (bbox['y2'] as num?)?.toDouble() ?? 0.0,
          'width': (bbox['width'] as num?)?.toDouble() ?? 0.0,
          'height': (bbox['height'] as num?)?.toDouble() ?? 0.0,
        };
      }
      detMap['coordinates'] = coords;
      beanDetections.add(detMap);
    }

    final Map<String, dynamic>? top =
        beanDetections.isNotEmpty ? Map<String, dynamic>.from(beanDetections.first) : null;
    final String topClass = (top?['class_name'] ?? top?['defect_type'] ?? 'Unknown').toString();
    final double topConf = (top?['confidence'] as num?)?.toDouble() ?? 0.0;

    // Map YOLO class to expected bean classes list (CoffeeNet order)
    const List<String> beanTypes = <String>['Liberica', 'Arabica', 'Robusta', 'Excelsa'];
    final Map<String, String> nameMap = {
      'arabica': 'Arabica',
      'robusta': 'Robusta',
      'liberica': 'Liberica',
      'excelsa': 'Excelsa',
    };
    final List<double> probabilityList = List<double>.filled(beanTypes.length, 0.0);
    final String normalizedTop = nameMap[topClass.toLowerCase()] ?? topClass;
    int mappedIdx = beanTypes.indexWhere(
      (name) => name.toLowerCase() == normalizedTop.toLowerCase(),
    );
    if (mappedIdx < 0 && beanTypes.isNotEmpty) {
      mappedIdx = 0;
    }
    if (mappedIdx >= 0 && mappedIdx < probabilityList.length) {
      probabilityList[mappedIdx] = topConf;
    }

    final Map<String, dynamic> prediction = {
      'predicted_class': mappedIdx >= 0 && mappedIdx < beanTypes.length ? beanTypes[mappedIdx] : normalizedTop,
      'confidence': topConf,
      'all_probabilities': probabilityList,
      'classes': beanTypes,
      'raw_class': topClass,
    };

    // Defect detections: prefer separate defect model output if provided
    final Map<String, dynamic> defectPayload =
        jsonResponse['defect'] is Map ? Map<String, dynamic>.from(jsonResponse['defect'] as Map) : const {};
    final List<dynamic> rawDefects = (defectPayload['detections'] as List?) ?? const [];
    final List<Map<String, dynamic>> defectDetections = [];
    final Map<String, int> defectTypes = {};

    if (rawDefects.isNotEmpty) {
      for (final det in rawDefects) {
        if (det is! Map) continue;
        final detMap = Map<String, dynamic>.from(det);
        final bbox = detMap['bbox'];
        Map<String, dynamic> coords = {};
        if (bbox is Map) {
          coords = {
            'x1': (bbox['x1'] as num?)?.toDouble() ?? 0.0,
            'y1': (bbox['y1'] as num?)?.toDouble() ?? 0.0,
            'x2': (bbox['x2'] as num?)?.toDouble() ?? 0.0,
            'y2': (bbox['y2'] as num?)?.toDouble() ?? 0.0,
            'width': (bbox['width'] as num?)?.toDouble() ?? 0.0,
            'height': (bbox['height'] as num?)?.toDouble() ?? 0.0,
          };
        }
        detMap['coordinates'] = coords;
        detMap['defect_type'] = detMap['class_name'] ?? detMap['defect_type'] ?? detMap['label'] ?? 'unknown';
        if (imageSize.isNotEmpty) {
          detMap['image_size'] = {
            'width': (imageSize['width'] as num?)?.toDouble(),
            'height': (imageSize['height'] as num?)?.toDouble(),
          };
          detMap['image_width'] = (imageSize['width'] as num?)?.toDouble();
          detMap['image_height'] = (imageSize['height'] as num?)?.toDouble();
        }
        String typeKey = detMap['defect_type'].toString();
        if (typeKey.isEmpty || typeKey.toLowerCase() == 'unknown') {
          typeKey = 'good_bean';
          detMap['defect_type'] = typeKey;
        }
        defectTypes[typeKey] = (defectTypes[typeKey] ?? 0) + 1;
        defectDetections.add(detMap);
      }
    } else {
      // Fallback: no separate defect model; reuse bean detections
      for (final detMap in beanDetections) {
        String typeKey = detMap['defect_type']?.toString() ??
            detMap['class_name']?.toString() ??
            'unknown';
        if (typeKey.isEmpty || typeKey.toLowerCase() == 'unknown') {
          typeKey = 'good_bean';
          detMap['defect_type'] = typeKey;
        }
        defectTypes[typeKey] = (defectTypes[typeKey] ?? 0) + 1;
        defectDetections.add(detMap);
      }
    }

    // Keep all detections (for masks/visuals), but treat "good_bean" as non-defect in scoring
    final List<Map<String, dynamic>> allDefects = defectDetections;
    final List<Map<String, dynamic>> filteredDefects = defectDetections
        .where((d) => (d['defect_type']?.toString().toLowerCase() ?? '') != 'good_bean')
        .toList();
    final Map<String, int> filteredTypes = {};
    for (final entry in defectTypes.entries) {
      if (entry.key.toLowerCase() == 'good_bean') continue;
      filteredTypes[entry.key] = entry.value;
    }

    const Map<String, double> defectWeights = {
      'broken_cut': 0.35,
      'fully_black': 0.4,
      'insect_damage': 0.3,
      'roasted_beans': 0.15,
    };

    double weightedScore = 0.0;
    for (final defect in filteredDefects) {
      final String key = defect['defect_type']?.toString().toLowerCase() ?? '';
      final double weight = defectWeights[key] ?? 0.1;
      final double conf = (defect['confidence'] as num?)?.toDouble() ?? 1.0;
      weightedScore += weight * conf;
    }

    final double defectPercentage = min(100.0, weightedScore * 20.0);
    // If no defects, treat as perfect score
    final double qualityScore =
        filteredDefects.isEmpty ? 1.0 : max(0.0, 1.0 - weightedScore * 0.2).toDouble();

    String grade(double score) {
      if (score >= 0.9) return 'A+';
      if (score >= 0.8) return 'A';
      if (score >= 0.7) return 'B+';
      if (score >= 0.6) return 'B';
      if (score >= 0.5) return 'C+';
      if (score >= 0.4) return 'C';
      if (score >= 0.3) return 'D';
      return 'F';
    }

    final Map<String, dynamic> defectDetection = {
      // Filtered for tables/metrics, but keep full list for overlays
      'detections': filteredDefects,
      'detections_all': allDefects,
      'summary': <String, dynamic>{
        'total_defects': filteredDefects.length,
        'defect_types': filteredTypes,
        'defect_percentage': defectPercentage,
        'quality_score': qualityScore,
        'quality_grade': grade(qualityScore),
      },
      'image_dimensions': {
        'width': (imageSize['width'] as num?)?.toDouble() ?? 0.0,
        'height': (imageSize['height'] as num?)?.toDouble() ?? 0.0,
      },
    };

    final Map<String, dynamic> summary =
        defectDetection['summary'] as Map<String, dynamic>;

    final Map<String, dynamic> healthScore = {
      'score': topConf,
      'percentage': topConf * 100.0,
      'grade': grade(topConf),
      'defect_count': filteredDefects.length,
    };

    final Map<String, dynamic> shelfLife = {
      'predicted_days': 180,
      'confidence_score': qualityScore,
      'category': 'estimated',
      'defect_score': summary['total_defects'],
      'defect_counts': filteredTypes,
      'defect_percentage': defectPercentage,
      'severity': summary['quality_grade'],
      'quality_grade': summary['quality_grade'],
      'base_shelf_life': 180,
    };

    return {
      'success': jsonResponse['success'] ?? true,
      'data': {
        'prediction': prediction,
        'defect_detection': defectDetection,
        'health_score': healthScore,
        'shelf_life': shelfLife,
        'model': jsonResponse['model'],
      },
    };
  }

  /// Detect defects only in a coffee bean image
  static Future<DefectDetection?> detectDefects(File imageFile, {double confidenceThreshold = 0.5}) async {
    try {
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiUrl/api/v1/detect-defects?confidence_threshold=$confidenceThreshold'),
      );

      // Add the image file
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
        ),
      );

      // Send the request
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonResponse = json.decode(response.body);
        return DefectDetection.fromJson(jsonResponse);
      } else {
        _logApiService('API Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e, stackTrace) {
      _logApiService(
        'Defect detection failed',
        error: e,
        stackTrace: stackTrace,
      );
      return null;
    }
  }

  /// Get defect detection with detailed error handling
  static Future<Map<String, dynamic>> detectDefectsWithErrorHandling(File imageFile, {double confidenceThreshold = 0.5}) async {
    try {
      // First check if API is available
      final isHealthy = await checkHealth();
      if (!isHealthy) {
        return {
          'success': false,
          'error': 'API server is not available. Please make sure the backend is running.',
        };
      }

      // Detect defects
      final defectDetection = await detectDefects(imageFile, confidenceThreshold: confidenceThreshold);
      if (defectDetection != null) {
        return {
          'success': true,
          'defect_detection': defectDetection,
        };
      } else {
        return {
          'success': false,
          'error': 'Failed to get defect detection from API',
        };
      }
    } catch (e, stackTrace) {
      _logApiService(
        'detectDefectsWithErrorHandling failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  /// Test scan endpoint to debug issues
  static Future<Map<String, dynamic>> testScanEndpoint(File imageFile) async {
    try {
      // First check if API is available
      final isHealthy = await checkHealth();
      if (!isHealthy) {
        return {
          'success': false,
          'error': 'API server is not available. Please make sure the backend is running.',
        };
      }

      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiUrl/api/v1/scan'),
      );

      // Add the image file
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
        ),
      );

      // Add optional device identifier
      final deviceId = await _getDeviceId();
      if (deviceId != null && deviceId.isNotEmpty) {
        request.fields['device_id'] = deviceId;
      }

      // Send the request
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonResponse = json.decode(response.body);
        return {
          'success': true,
          'data': jsonResponse,
        };
      } else {
        _logApiService('API Error: ${response.statusCode} - ${response.body}');
        return {
          'success': false,
          'error': 'API Error: ${response.statusCode} - ${response.body}',
        };
      }
    } catch (e, stackTrace) {
      _logApiService(
        'Test scan failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  // ===================== History Endpoints =====================
  static Future<Map<String, dynamic>> fetchHistory({int limit = 50, int offset = 0}) async {
    try {
      // Ensure base URL is reachable and resolved
      await checkHealth();
      final deviceId = await _getDeviceId();
      _logApiService('=== HISTORY FETCH DEBUG ===');
      _logApiService('Device ID for history: $deviceId');
      final url = Uri.parse('$apiUrl/api/v1/history?device_id=${Uri.encodeComponent(deviceId ?? '')}&limit=$limit&offset=$offset');
      // Debug: log URL
      _logApiService('Fetching history: GET $url');
      _logApiService('=== END HISTORY DEBUG ===');
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return {'success': true, 'data': json.decode(response.body)};
      }
      _logApiService('History API Error: ${response.statusCode} - ${response.body}');
      return {'success': false, 'error': 'API Error ${response.statusCode}: ${response.body}'};
    } catch (e, stackTrace) {
      _logApiService(
        'History fetch failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  static Future<Map<String, dynamic>> fetchHistoryDetails(int historyId) async {
    try {
      await checkHealth();
      final url = Uri.parse('$apiUrl/api/v1/history/$historyId');
      _logApiService('Fetching history details: GET $url');
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return {'success': true, 'data': json.decode(response.body)};
      }
      _logApiService('History details API Error: ${response.statusCode} - ${response.body}');
      return {'success': false, 'error': 'API Error ${response.statusCode}: ${response.body}'};
    } catch (e, stackTrace) {
      _logApiService(
        'History details fetch failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  // Persisted per-install device ID (no auth)
  static String? _cachedDeviceId;
  static Future<String?> _getDeviceId() async {
    try {
      if (_cachedDeviceId != null) {
        _logApiService('Using cached device ID: $_cachedDeviceId');
        return _cachedDeviceId;
      }
      // Use a simple on-disk GUID stored in app documents directory
      final dir = await _getAppDir();
      final file = File('${dir.path}/beanscan_device_id.txt');
      if (await file.exists()) {
        final id = (await file.readAsString()).trim();
        if (id.isNotEmpty) {
          _cachedDeviceId = id;
          _logApiService('Loaded existing device ID: $id');
          return id;
        }
      }
      final newId = _generateGuid();
      await file.writeAsString(newId, flush: true);
      _cachedDeviceId = newId;
      _logApiService('Generated new device ID: $newId');
      return newId;
    } catch (e, stackTrace) {
      _logApiService(
        'Error getting device ID',
        error: e,
        stackTrace: stackTrace,
      );
      return null;
    }
  }

  // Force regenerate device ID (useful for testing or when switching devices)
  static Future<String?> regenerateDeviceId() async {
    try {
      final dir = await _getAppDir();
      final file = File('${dir.path}/beanscan_device_id.txt');
      final newId = _generateGuid();
      await file.writeAsString(newId, flush: true);
      _cachedDeviceId = newId;
      _logApiService('Force regenerated device ID: $newId');
      return newId;
    } catch (e, stackTrace) {
      _logApiService(
        'Error regenerating device ID',
        error: e,
        stackTrace: stackTrace,
      );
      return null;
    }
  }

  // Test function to check if backend is working and create a test user
  static Future<Map<String, dynamic>> testBackendConnection() async {
    try {
      final deviceId = await _getDeviceId();
      _logApiService('=== BACKEND CONNECTION TEST ===');
      _logApiService('Device ID: $deviceId');
      
      // Test health endpoint
      final healthResult = await checkHealth();
      _logApiService('Health check: $healthResult');
      
      // Test history endpoint
      final historyResult = await fetchHistory(limit: 5);
      _logApiService('History test result: $historyResult');
      
      return {
        'success': true,
        'device_id': deviceId,
        'health_check': healthResult,
        'history_test': historyResult
      };
    } catch (e, stackTrace) {
      _logApiService(
        'Backend connection test failed',
        error: e,
        stackTrace: stackTrace,
      );
      return {
        'success': false,
        'error': e.toString()
      };
    }
  }

  static Future<Directory> _getAppDir() async {
    final dir = await getApplicationSupportDirectory();
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static String _generateGuid() {
    // Random v4 style GUID
    final rnd = Random.secure();
    List<int> bytes(int length) => List<int>.generate(length, (_) => rnd.nextInt(256));
    String hex(List<int> b) => b.map((v) => v.toRadixString(16).padLeft(2, '0')).join();
    final b = bytes(16);
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant
    final s = hex(b);
    return '${s.substring(0,8)}-${s.substring(8,12)}-${s.substring(12,16)}-${s.substring(16,20)}-${s.substring(20,32)}';
  }
}
