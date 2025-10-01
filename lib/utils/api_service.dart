import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

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
    return BeanPrediction(
      prediction: json['prediction'] ?? '',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      allProbabilities: Map<String, double>.from(
        json['all_probabilities'] ?? {},
      ),
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
    if (Platform.isAndroid) return androidBaseUrl;
    return baseUrl;
  }

  /// Check if the API is healthy
  static Future<bool> checkHealth() async {
    final candidates = <String>[
      apiUrl,
      if (Platform.isAndroid) ...[
        // Android emulator default host mapping
        'http://10.0.2.2:8000',
        // Genymotion emulator
        'http://10.0.3.2:8000',
      ],
      // Common local fallbacks
      'http://localhost:8000',
      'http://127.0.0.1:8000',
    ];

    for (final url in candidates) {
      try {
        final response = await http.get(
          Uri.parse('$url/health'),
          headers: {'Content-Type': 'application/json'},
        ).timeout(const Duration(seconds: 8));
        if (response.statusCode == 200) {
          _resolvedApiUrl = url;
          if (url != apiUrl) {
            print('API reachable at: $url (selected)');
          }
          return true;
        }
      } catch (e) {
        // Try next candidate
        print('Health check failed for $url: $e');
      }
    }
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
    } catch (e) {
      print('Prediction failed: $e');
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
    } catch (e) {
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  static Future<http.StreamedResponse> _sendWithTimeoutAndRetry(http.MultipartRequest request, {int retries = 1, Duration timeout = const Duration(seconds: 60)}) async {
    int attempt = 0;
    while (true) {
      attempt += 1;
      try {
        final streamed = await request.send().timeout(timeout);
        return streamed;
      } catch (e) {
        if (attempt > retries) rethrow;
        await Future.delayed(Duration(seconds: 2 * attempt));
      }
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

      // Send the request with timeout and single retry to mitigate cold starts/502
      final streamedResponse = await _sendWithTimeoutAndRetry(request, retries: 1, timeout: const Duration(seconds: 90));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonResponse = json.decode(response.body);
        return {
          'success': true,
          'data': jsonResponse,
        };
      } else {
        print('API Error: ${response.statusCode} - ${response.body}');
        return {
          'success': false,
          'error': 'API Error: ${response.statusCode} - ${response.body}',
        };
      }
    } catch (e) {
      print('Scan failed: $e');
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
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

      // Send the request with timeout
      final streamedResponse = await _sendWithTimeoutAndRetry(request, retries: 1, timeout: const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonResponse = json.decode(response.body);
        return DefectDetection.fromJson(jsonResponse);
      } else {
        print('API Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('Defect detection failed: $e');
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
    } catch (e) {
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

      final streamedResponse = await _sendWithTimeoutAndRetry(request, retries: 1, timeout: const Duration(seconds: 90));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonResponse = json.decode(response.body);
        return {
          'success': true,
          'data': jsonResponse,
        };
      } else {
        print('API Error: ${response.statusCode} - ${response.body}');
        return {
          'success': false,
          'error': 'API Error: ${response.statusCode} - ${response.body}',
        };
      }
    } catch (e) {
      print('Test scan failed: $e');
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
      final url = Uri.parse('$apiUrl/api/v1/history?device_id=${Uri.encodeComponent(deviceId ?? '')}&limit=$limit&offset=$offset');
      // ignore: avoid_print
      print('Fetching history: GET ' + url.toString());
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return {'success': true, 'data': json.decode(response.body)};
      }
      // ignore: avoid_print
      print('History API Error: ' + response.statusCode.toString() + ' - ' + response.body);
      return {'success': false, 'error': 'API Error ${response.statusCode}: ${response.body}'};
    } catch (e) {
      // ignore: avoid_print
      print('History fetch failed: ' + e.toString());
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  static Future<Map<String, dynamic>> fetchHistoryDetails(int historyId) async {
    try {
      await checkHealth();
      final url = Uri.parse('$apiUrl/api/v1/history/$historyId');
      // ignore: avoid_print
      print('Fetching history details: GET ' + url.toString());
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return {'success': true, 'data': json.decode(response.body)};
      }
      // ignore: avoid_print
      print('History details API Error: ' + response.statusCode.toString() + ' - ' + response.body);
      return {'success': false, 'error': 'API Error ${response.statusCode}: ${response.body}'};
    } catch (e) {
      // ignore: avoid_print
      print('History details fetch failed: ' + e.toString());
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  // Persisted per-install device ID (no auth)
  static String? _cachedDeviceId;
  static Future<String?> _getDeviceId() async {
    try {
      if (_cachedDeviceId != null) return _cachedDeviceId;
      // Use a simple on-disk GUID stored in app documents directory
      final dir = await _getAppDir();
      final file = File('${dir.path}/beanscan_device_id.txt');
      if (await file.exists()) {
        final id = (await file.readAsString()).trim();
        if (id.isNotEmpty) {
          _cachedDeviceId = id;
          return id;
        }
      }
      final newId = _generateGuid();
      await file.writeAsString(newId, flush: true);
      _cachedDeviceId = newId;
      return newId;
    } catch (_) {
      return null;
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
