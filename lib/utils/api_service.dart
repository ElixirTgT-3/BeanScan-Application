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

class ApiService {
  static const String baseUrl = 'http://localhost:8000';
  
  // For Android device, use computer's actual IP address
  static const String androidBaseUrl = 'http://192.168.1.7:8000';
  
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

  /// Predict bean type from image file
  static Future<BeanPrediction?> predictBeanType(File imageFile) async {
    try {
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiUrl/api/predict'),
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
        return BeanPrediction.fromJson(jsonResponse);
      } else {
        print('API Error: ${response.statusCode} - ${response.body}');
        return null;
      }
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
}
