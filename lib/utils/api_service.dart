import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

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
  static const String baseUrl = 'http://localhost:8000';
  
  // For Android device, use computer's actual IP address
  static const String androidBaseUrl = 'http://192.168.1.9:8000';
  
  static String get apiUrl {
    // Check if running on Android emulator
    if (Platform.isAndroid) {
      return androidBaseUrl;
    }
    return baseUrl;
  }

  /// Check if the API is healthy
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$apiUrl/health'),
        headers: {'Content-Type': 'application/json'},
      );
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
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

      // Send the request
      final streamedResponse = await request.send();
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
        Uri.parse('$apiUrl/api/v1/test-scan'),
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
}
