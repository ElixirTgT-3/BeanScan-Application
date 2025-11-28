import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';
import '../utils/local_history_store.dart';
import '../utils/app_settings.dart';
import '../utils/app_logger.dart';
import '../utils/demo_override.dart';
import 'results_page.dart';

// Custom painter for sun icon with 8 rays
class _SunIconPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.orange
      ..style = PaintingStyle.fill;
    
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 4; // Circle radius
    
    // Draw the central circle
    canvas.drawCircle(center, radius, paint);
    
    // Draw 8 rays evenly spaced around the circle
    final rayLength = 6.0;
    final rayWidth = 2.0;
    final rayDistance = radius + 2; // Distance from center to start of ray
    
    for (int i = 0; i < 8; i++) {
      final angle = (i * math.pi / 4); // 8 rays = 45 degrees apart
      final startX = center.dx + math.cos(angle) * rayDistance;
      final startY = center.dy + math.sin(angle) * rayDistance;
      final endX = center.dx + math.cos(angle) * (rayDistance + rayLength);
      final endY = center.dy + math.sin(angle) * (rayDistance + rayLength);
      
      canvas.drawLine(
        Offset(startX, startY),
        Offset(endX, endY),
        paint..strokeWidth = rayWidth..strokeCap = StrokeCap.round,
      );
    }
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _OverrideShapeOption {
  final String beanType;
  final IconData icon;

  const _OverrideShapeOption({
    required this.beanType,
    required this.icon,
  });
}

const List<_OverrideShapeOption> _overrideShapeOrder = [
  _OverrideShapeOption(beanType: 'Excelsa', icon: Icons.crop_square_rounded),
  _OverrideShapeOption(beanType: 'Arabica', icon: Icons.egg_outlined),
  _OverrideShapeOption(beanType: 'Liberica', icon: Icons.change_history),
  _OverrideShapeOption(beanType: 'Robusta', icon: Icons.circle_outlined),
];

void _logScanPage(
  String message, {
  Object? error,
  StackTrace? stackTrace,
}) {
  logDebug(
    'ScanPage',
    message,
    error: error,
    stackTrace: stackTrace,
  );
}

class ScanPage extends StatefulWidget {
  final VoidCallback? onClose;
  const ScanPage({super.key, this.onClose});

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;
  bool _isCameraInitialized = false;
  bool _isFlashOn = false;
  bool _isPermissionGranted = false;
  // Zoom state
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  double _currentZoom = 1.0;
  double _gestureBaseZoom = 1.0;
  // Exposure (brightness) state
  double _minExposure = 0.0;
  double _maxExposure = 0.0;
  double _currentExposure = 0.0;
  bool _showBrightnessControl = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Check permissions immediately on initialization
    _checkPermissionsOnInit();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      if (_cameraController != null && _cameraController!.value.isInitialized) {
        _cameraController?.dispose();
      }
    } else if (state == AppLifecycleState.resumed) {
      // Check permissions again when app resumes (user might have changed them in settings)
      _checkPermissionsOnInit();
    }
  }

  Future<void> _checkPermissionsOnInit() async {
    try {
      debugPrint('=== Checking permissions on initialization ===');
      
      // Check current permission status without requesting
      PermissionStatus cameraStatus = await Permission.camera.status;
      PermissionStatus storageStatus = await Permission.storage.status;
      
      debugPrint('Initial camera permission status: $cameraStatus');
      debugPrint('Initial storage permission status: $storageStatus');
      
      // If permissions are already granted, initialize camera immediately
      if (cameraStatus.isGranted && storageStatus.isGranted) {
        debugPrint('All permissions already granted, initializing camera...');
        setState(() {
          _isPermissionGranted = true;
        });
        _initializeCamera();
      } else {
        debugPrint('Permissions not granted, showing permission request UI');
        setState(() {
          _isPermissionGranted = false;
        });
      }
    } catch (e) {
      debugPrint('Error checking permissions on init: $e');
      setState(() {
        _isPermissionGranted = false;
      });
    }
  }

  Future<void> _checkPermissions() async {
    try {
      debugPrint('=== Starting permission check ===');
      
      // Check current permission status first
      PermissionStatus cameraStatus = await Permission.camera.status;
      PermissionStatus storageStatus = await Permission.storage.status;
      
      debugPrint('Initial camera permission status: $cameraStatus');
      debugPrint('Initial storage permission status: $storageStatus');
      
      // If permissions are not granted, request them
      if (!cameraStatus.isGranted) {
        debugPrint('Requesting camera permission...');
        
        // Try to request permission
        cameraStatus = await Permission.camera.request();
        debugPrint('Camera permission request result: $cameraStatus');
        
        // If still not granted, check if it's permanently denied
        if (!cameraStatus.isGranted) {
          cameraStatus = await Permission.camera.status;
          debugPrint('Final camera permission status: $cameraStatus');
        }
      }
      
      if (!storageStatus.isGranted) {
        debugPrint('Requesting storage permission...');
        storageStatus = await Permission.storage.request();
        debugPrint('Storage permission request result: $storageStatus');
        
        // Wait a moment for the permission dialog to complete
        await Future.delayed(const Duration(milliseconds: 1000));
        
        // Check the status again after the request
        storageStatus = await Permission.storage.status;
        debugPrint('Storage permission status after request: $storageStatus');
      }
      
      debugPrint('=== Final permission status ===');
      debugPrint('Camera: $cameraStatus');
      debugPrint('Storage: $storageStatus');
      
      // Check if permissions are now granted
      if (cameraStatus.isGranted && storageStatus.isGranted) {
        debugPrint('All permissions granted, initializing camera...');
        setState(() {
          _isPermissionGranted = true;
        });
        _initializeCamera();
      } else {
        setState(() {
          _isPermissionGranted = false;
        });
        
        // Show more detailed feedback about what permissions are missing
        if (mounted) {
          List<String> missingPermissions = [];
          if (!cameraStatus.isGranted) missingPermissions.add('Camera');
          if (!storageStatus.isGranted) missingPermissions.add('Storage');
          
          String message = '${missingPermissions.join(' and ')} permission${missingPermissions.length > 1 ? 's' : ''} required';
          
          if (cameraStatus.isPermanentlyDenied || storageStatus.isPermanentlyDenied) {
            message += ' - Please enable in app settings';
          }
          
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(message),
              backgroundColor: Colors.orange,
              duration: const Duration(seconds: 4),
              action: SnackBarAction(
                label: 'Settings',
                onPressed: _openAppSettings,
                textColor: Colors.white,
              ),
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Error checking permissions: $e');
      setState(() {
        _isPermissionGranted = false;
      });
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        debugPrint('No cameras available');
        return;
      }

      _cameraController = CameraController(
        _cameras[_selectedCameraIndex],
        ResolutionPreset.medium, // Use medium for better performance
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _cameraController!.initialize();
      // Query camera capabilities for zoom and exposure ranges
      try {
        _minZoom = await _cameraController!.getMinZoomLevel();
        _maxZoom = await _cameraController!.getMaxZoomLevel();
        _currentZoom = _currentZoom.clamp(_minZoom, _maxZoom);
      } catch (_) {}
      try {
        _minExposure = await _cameraController!.getMinExposureOffset();
        _maxExposure = await _cameraController!.getMaxExposureOffset();
        _currentExposure = _currentExposure.clamp(_minExposure, _maxExposure);
        await _cameraController!.setExposureOffset(_currentExposure);
      } catch (_) {}
      
      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Camera initialization failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    final controller = _cameraController!;
    late final XFile capturedImage;
    try {
      // Ensure flash is used if it's supposed to be on
      if (_isFlashOn) {
        debugPrint('Taking picture with flash ON');
        await controller.setFlashMode(FlashMode.torch);
        await Future.delayed(const Duration(milliseconds: 300)); // Longer delay for flash to activate
      } else {
        debugPrint('Taking picture with flash OFF');
        await controller.setFlashMode(FlashMode.off);
      }
      
      capturedImage = await controller.takePicture();
      debugPrint('Picture taken: ${capturedImage.path}');
    } catch (e) {
      debugPrint('Error taking picture: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Failed to take picture'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return;
    } finally {
      if (controller.value.isInitialized) {
        try {
          await controller.setFlashMode(FlashMode.off);
        } catch (flashError) {
          debugPrint('Failed to reset flash: $flashError');
        }
      }
      if (_isFlashOn) {
        if (mounted) {
          setState(() {
            _isFlashOn = false;
          });
        } else {
          _isFlashOn = false;
        }
      }
    }

    // Show loading indicator
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Analyzing image...'),
          backgroundColor: Colors.blue,
          duration: Duration(seconds: 2),
        ),
      );
    }
    
    // Basic validation: reject overly dark images before sending
    final file = File(capturedImage.path);
    if (await _isImageTooDark(file) || !(await _hasEnoughBrownPixels(file))) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Please upload a clear, well-lit photo of coffee beans.'),
            backgroundColor: Colors.orange,
          ),
        );
      }
      // Continue anyway; final decision will be made after model prediction
    }

    // Process the image with API
    await _processImage(file, fromGallery: false);
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2) return;

    _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras.length;
    await _cameraController?.dispose();
    await _initializeCamera();
  }

  Future<void> _toggleFlash() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final newFlashMode = _isFlashOn ? FlashMode.off : FlashMode.torch;
      debugPrint('Toggling flash from ${_isFlashOn ? 'on' : 'off'} to ${newFlashMode == FlashMode.torch ? 'on' : 'off'}');
      
      await _cameraController!.setFlashMode(newFlashMode);
      
      // Wait for flash mode to be set
      await Future.delayed(const Duration(milliseconds: 200));
      
      setState(() {
        _isFlashOn = !_isFlashOn;
      });
      
      debugPrint('Flash toggled successfully: $_isFlashOn');
      
      // Show user feedback
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(_isFlashOn ? 'Flash turned ON' : 'Flash turned OFF'),
            backgroundColor: _isFlashOn ? Colors.green : Colors.grey,
            duration: const Duration(seconds: 1),
          ),
        );
      }
    } catch (e) {
      debugPrint('Error toggling flash: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Flash not supported: ${e.toString()}'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    }
  }

  void _toggleBrightnessControlVisibility() {
    _showBrightnessControl = !_showBrightnessControl;
    if (mounted) {
      setState(() {});
    }
  }

  void _cycleOverrideSelection() {
    final demo = DemoOverride();
    if (!demo.isEnabled) return;

    final current = demo.overrideBeanType;
    int nextIndex = 0;
    if (current != null) {
      final currentIndex = _overrideShapeOrder.indexWhere(
        (option) => option.beanType.toLowerCase() == current.toLowerCase(),
      );
      nextIndex = currentIndex == -1
          ? 0
          : (currentIndex + 1) % _overrideShapeOrder.length;
    }
    demo.setOverride(_overrideShapeOrder[nextIndex].beanType);
    if (mounted) {
      setState(() {});
    }
  }

  void _handleCameraTap() {
    final demo = DemoOverride();
    if (demo.isEnabled) {
      _cycleOverrideSelection();
    } else {
      _toggleBrightnessControlVisibility();
    }
  }

  Future<void> _processImage(File imageFile, {required bool fromGallery}) async {
    try {
      final demo = DemoOverride();
      
      // Show loading dialog with demo override capability
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext dialogContext) {
          return StatefulBuilder(
            builder: (context, setDialogState) {
              return DemoOverrideLoadingOverlay(
                child: AlertDialog(
                  content: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: const [
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(width: 20),
                          Expanded(
                            child: Text("Analyzing bean type and detecting defects..."),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              );
            },
          );
        },
      );

      // Test the API connection first
      final testResult = await ApiService.scanBeanImage(imageFile);
      if (!testResult['success']) {
        if (mounted) {
          Navigator.of(context).pop();
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Scan failed: ${testResult['error']}'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }
      
      // Hide loading dialog
      if (mounted) {
        Navigator.of(context).pop();
      }

      if (testResult['success'] && testResult['data'] != null) {
        // Debug: Print the full API response
        _logScanPage('API Response Debug:');
        _logScanPage('  - testResult: $testResult');
        _logScanPage('  - data: ${testResult['data']}');
        
        final healthScoreData = testResult['data']['data']['health_score'];
        final double healthScorePercentage = (healthScoreData?['percentage'] as num?)?.toDouble() ?? 0.0;
        final double derivedConfidence = ((healthScorePercentage / 100).clamp(0.0, 1.0)).toDouble();

        // Convert the prediction data to BeanPrediction object
        // Apply demo override if enabled
        Map<String, dynamic> predictionData = Map<String, dynamic>.from(
          testResult['data']['data']['prediction'] as Map
        );
        
        // Apply demo override if set
        if (demo.isEnabled) {
          if (demo.overrideBeanType == null) {
            demo.initFromPrediction(predictionData['predicted_class']?.toString() ?? '');
          }
          predictionData = demo.applyOverride(predictionData);
          _logScanPage('  - Demo override applied: ${demo.overrideBeanType}');
        }
        
        _logScanPage('  - predictionData: $predictionData');

        // Validate that the image looks like coffee beans
        String predictedClass = (predictionData['predicted_class'] ?? '').toString();
        double predictedConfidence = (predictionData['confidence'] ?? 0.0).toDouble();
        const List<String> beanTypes = kBeanTypes;
        final List<dynamic> probabilitiesRaw =
            predictionData['all_probabilities'] as List<dynamic>? ?? const <dynamic>[];
        final List<double> probabilityValues = [
          for (int i = 0; i < probabilitiesRaw.length && i < beanTypes.length; i++)
            (probabilitiesRaw[i] as num).toDouble(),
        ];

        final _ImageHeuristics heuristics = await _analyzeImageHeuristics(imageFile);
        final double brownRatio = heuristics.brownRatio;
        final double textureScore = heuristics.textureScore;
        final double contrastScore = heuristics.contrast;
        final double visualScore = _computeVisualBeanScore(
          brownRatio: brownRatio,
          textureScore: textureScore,
          contrastScore: contrastScore,
        );

        bool isKnownBean = beanTypes.contains(predictedClass);
        if (!isKnownBean && probabilityValues.isNotEmpty) {
          final int topIndex = _indexOfMax(probabilityValues);
          if (topIndex != -1) {
            final double topProbability = probabilityValues[topIndex];
            final bool heuristicsSupport = visualScore >= 0.3;
            if (topProbability >= 0.28 && heuristicsSupport && topIndex < beanTypes.length) {
              predictedClass = beanTypes[topIndex];
              predictedConfidence = topProbability;
              isKnownBean = true;
              _logScanPage(
                'Applied fallback bean label: $predictedClass (prob=${(topProbability * 100).toStringAsFixed(1)}%)',
              );
            }
          }
        }

        final double bestConfidence =
            predictedConfidence > derivedConfidence ? predictedConfidence : derivedConfidence;
        final double normalizedConfidence = math.min(1.0, math.max(0.0, bestConfidence));
        const double minConfidenceThreshold = 0.45;
        const double minBrownRatioThreshold = 0.012;
        const double criticalBrownThreshold = 0.005;
        const double minTextureThreshold = 0.045;
        const double minContrastThreshold = 7.0;
        final bool heuristicsVeryWeak = visualScore < 0.12;
        final bool heuristicsStrong = visualScore >= 0.38;

        final List<String> rejectionReasons = [];
        final bool failClass = !isKnownBean && !heuristicsStrong;
        final bool failConfidence = normalizedConfidence < minConfidenceThreshold && !heuristicsStrong;
        final bool failColor = brownRatio < minBrownRatioThreshold && !heuristicsStrong && normalizedConfidence < 0.6;
        final bool failColorCritical = brownRatio < criticalBrownThreshold && !heuristicsStrong;
        final bool failTexture = textureScore < minTextureThreshold && !heuristicsStrong && normalizedConfidence < 0.55;
        final bool failContrast = contrastScore < minContrastThreshold && !heuristicsStrong && normalizedConfidence < 0.55;
        final bool failVisualScore = heuristicsVeryWeak && normalizedConfidence < 0.5;

        if (failClass) {
          final displayClass = predictedClass.isEmpty ? 'another object' : predictedClass;
          rejectionReasons.add(
            'The AI labeled this photo as "$displayClass", which is not a supported coffee bean type.',
          );
        }
        if (failConfidence) {
          rejectionReasons.add(
            'The AI is only ${(normalizedConfidence * 100).clamp(0, 100).toStringAsFixed(0)}% sure the image shows coffee beans.',
          );
        }
        if (failColor) {
          rejectionReasons.add(
            'Only ${(brownRatio * 100).clamp(0, 100).toStringAsFixed(1)}% of the pixels match typical coffee bean colors.',
          );
        }
        if (failTexture) {
          rejectionReasons.add(
            'The photo looks very smooth (texture score ${(textureScore * 100).toStringAsFixed(0)}), while coffee beans have more surface detail.',
          );
        }
        if (failContrast) {
          rejectionReasons.add(
            'Lighting/contrast is very low (contrast score ${contrastScore.toStringAsFixed(1)}); beans need sharper highlights and shadows.',
          );
        }

        if (failVisualScore) {
          rejectionReasons.add(
            'The photo lacks the bean-like colors and surface texture we expect (visual score ${(visualScore * 100).clamp(0, 100).toStringAsFixed(0)}%).',
          );
        }

        final int softFailCount = [
          failConfidence,
          failColor,
          failTexture,
          failContrast,
          failVisualScore,
        ].where((v) => v).length;

        final bool shouldReject = failClass ||
            failColorCritical ||
            failVisualScore ||
            (failColor && (failConfidence || failTexture || failContrast)) ||
            softFailCount >= 3;

        _logScanPage(
          'Heuristic check -> class=$predictedClass, confidence=${(normalizedConfidence * 100).toStringAsFixed(1)}%, '
          'brownRatio=${(brownRatio * 100).toStringAsFixed(2)}%, texture=${(textureScore * 100).toStringAsFixed(1)}%, '
          'contrast=${contrastScore.toStringAsFixed(1)}, visual=${(visualScore * 100).toStringAsFixed(0)}%, reject=$shouldReject',
        );

        if (shouldReject) {
          if (rejectionReasons.isEmpty) {
            rejectionReasons.add(
              'We could not verify enough visual cues that the image contains coffee beans.',
            );
          }
          await _showNonCoffeeDialog(
            fromGallery: fromGallery,
            reasons: rejectionReasons,
          );
          return;
        }
        
        final allProbabilities = <String, double>{};

        for (int i = 0; i < probabilityValues.length && i < beanTypes.length; i++) {
          allProbabilities[beanTypes[i]] = probabilityValues[i];
        }

        final beanPrediction = BeanPrediction(
          prediction: predictedClass,
          confidence: normalizedConfidence,
          allProbabilities: allProbabilities,
        );
        
        _logScanPage('  - beanPrediction: $beanPrediction');
        
        // Use the image URL directly from the scan response
        String imagePathToShow = imageFile.path; // Fallback to local file
        try {
          final imageUrl = testResult['data']['image_url'];
          if (imageUrl is String && imageUrl.isNotEmpty) {
            imagePathToShow = imageUrl;
            _logScanPage('Using backend image URL: $imagePathToShow');
          } else {
            _logScanPage('No image URL in response, using local file: $imagePathToShow');
          }
        } catch (e, stackTrace) {
          _logScanPage(
            'Could not get image URL from response',
            error: e,
            stackTrace: stackTrace,
          );
        }

        final double cachedHealthyPercent = healthScorePercentage.clamp(0.0, 100.0);
        final double cachedDefectivePercent = (100.0 - cachedHealthyPercent).clamp(0.0, 100.0);
        try {
          final localEntry = CachedHistoryEntry.fromScanResponse(
            response: testResult['data'] as Map<String, dynamic>,
            prediction: beanPrediction,
            healthyPercent: cachedHealthyPercent,
            defectivePercent: cachedDefectivePercent,
            imagePath: imagePathToShow,
            defectDetection: testResult['data']['data']['defect_detection'] as Map<String, dynamic>?,
            shelfLife: testResult['data']['data']['shelf_life'] as Map<String, dynamic>?,
          );
          await LocalHistoryStore.addEntry(localEntry);
        } catch (cacheErr, stackTrace) {
          _logScanPage(
            'Failed to cache history entry',
            error: cacheErr,
            stackTrace: stackTrace,
          );
        }

        // Navigate to results page with both classification and defect detection
        _logScanPage('About to navigate to ResultsPage...');
        if (mounted) {
          _logScanPage('Navigating to ResultsPage...');
          final result = await Navigator.of(context).push<ResultsNavigationAction?>(
            MaterialPageRoute(
              builder: (context) => ResultsPage(
                prediction: beanPrediction,
                defectDetection: testResult['data']['data']['defect_detection'],
                shelfLife: testResult['data']['data']['shelf_life'],
                imagePath: imagePathToShow,
                shouldAutoSave: AppSettings.instance.autoSaveScans,
              ),
            ),
          );
          _logScanPage('Navigation completed with result: $result');
          if (!mounted) {
            return;
          }
          if (result == ResultsNavigationAction.history) {
            if (widget.onClose != null) {
              widget.onClose!();
            } else if (Navigator.of(context).canPop()) {
              Navigator.of(context).pop();
            }
          }
        } else {
          _logScanPage('Widget not mounted, cannot navigate');
        }
      } else {
        // Show error message
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(testResult['error'] ?? 'Failed to analyze image'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e, stackTrace) {
      _logScanPage(
        'Failed to process image',
        error: e,
        stackTrace: stackTrace,
      );
      // Hide loading dialog
      if (mounted) {
        Navigator.of(context).pop();
      }
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _pickImageFromGallery() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: ImageSource.gallery);
      
      if (image != null) {
        final file = File(image.path);
        if (await _isImageTooDark(file) || !(await _hasEnoughBrownPixels(file))) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Please upload a clear, well‑lit photo of coffee beans.'),
                backgroundColor: Colors.orange,
              ),
            );
          }
          // Continue to model prediction
        }
        // Show loading indicator
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Analyzing image...'),
              backgroundColor: Colors.blue,
              duration: Duration(seconds: 2),
            ),
          );
        }
        
        // Process the image with API
        await _processImage(file, fromGallery: true);
      }
    } catch (e) {
      debugPrint('Error picking image: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error picking image: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _openAppSettings() async {
    try {
      debugPrint('Attempting to open app settings...');
      final result = await openAppSettings();
      debugPrint('App settings result: $result');
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('Please enable camera and storage permissions in settings, then restart the app'),
            backgroundColor: Colors.blue,
            duration: const Duration(seconds: 5),
            action: SnackBarAction(
              label: 'OK',
              onPressed: () {},
              textColor: Colors.white,
            ),
          ),
        );
      }
    } catch (e) {
      debugPrint('Error opening app settings: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    }
  }

  Future<void> _showNonCoffeeDialog({
    required bool fromGallery,
    required List<String> reasons,
  }) async {
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (ctx) {
        final uploadLabel = fromGallery ? 'Upload Different Image' : 'Upload Another Image';
        return AlertDialog(
          title: const Text('Try Again'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                "We couldn't confirm that this photo contains coffee beans.",
              ),
              if (reasons.isNotEmpty) ...[
                const SizedBox(height: 12),
                ...reasons.map(
                  (reason) => Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('- '),
                        Expanded(child: Text(reason)),
                      ],
                    ),
                  ),
                ),
              ],
              const SizedBox(height: 12),
              const Text(
                'Try again with beans filling most of the frame, good lighting, and minimal background.',
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(),
              child: const Text('Retake Photo'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(ctx).pop();
                if (mounted) {
                  Future.microtask(_pickImageFromGallery);
                }
              },
              child: Text(uploadLabel),
            ),
          ],
        );
      },
    );
  }

  // Quick luminance check to reject completely dark/blank images
  Future<bool> _isImageTooDark(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes, targetWidth: 32, targetHeight: 32);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) return false;
      final data = byteData.buffer.asUint8List();

      int sumLuma = 0;
      int count = 0;
      for (int i = 0; i < data.length; i += 4) {
        final r = data[i];
        final g = data[i + 1];
        final b = data[i + 2];
        // Perceived luminance
        final luma = (0.299 * r + 0.587 * g + 0.114 * b).round();
        sumLuma += luma;
        count++;
      }
      final avg = sumLuma / count;
      return avg < 18; // very dark threshold
    } catch (_) {
      return false;
    }
  }

  // Heuristic: require a minimum proportion of brownish pixels typical of roasted beans
  Future<bool> _hasEnoughBrownPixels(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes, targetWidth: 48, targetHeight: 48);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) return true; // don't block on failure
      final data = byteData.buffer.asUint8List();

      int brownish = 0;
      int total = 0;
      for (int i = 0; i < data.length; i += 4) {
        final r = data[i].toDouble();
        final g = data[i + 1].toDouble();
        final b = data[i + 2].toDouble();
        final brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b);
        // HSV-like heuristic: brown ~ low blue, moderate red/green, medium-low brightness
        final isBrown = r > 60 && g > 40 && b < 80 && r >= g && brightness > 40 && brightness < 160;
        if (isBrown) brownish++;
        total++;
      }
      final ratio = brownish / total;
      return ratio > 0.02; // minimal precheck; final gate uses _analyzeImageHeuristics
    } catch (_) {
      return true;
    }
  }

  Future<_ImageHeuristics> _analyzeImageHeuristics(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes, targetWidth: 96, targetHeight: 96);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final width = image.width;
      final height = image.height;
      final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      image.dispose();
      if (byteData == null || width == 0 || height == 0) {
        return const _ImageHeuristics();
      }
      final data = byteData.buffer.asUint8List();

      int brownish = 0;
      int total = 0;
      double brightnessSum = 0.0;
      double brightnessSqSum = 0.0;
      double textureAccumulator = 0.0;
      int neighborSamples = 0;

      double computeBrightness(double r, double g, double b) =>
          (0.2126 * r + 0.7152 * g + 0.0722 * b);

      for (int i = 0; i < data.length; i += 4) {
        final r = data[i].toDouble();
        final g = data[i + 1].toDouble();
        final b = data[i + 2].toDouble();
        final brightness = computeBrightness(r, g, b);
        final isBrown = r > 70 && g > 50 && b < 110 && r >= g && brightness > 35 && brightness < 200;
        if (isBrown) brownish++;
        total++;
        brightnessSum += brightness;
        brightnessSqSum += brightness * brightness;

        final int pixelIndex = i ~/ 4;
        final int x = pixelIndex % width;
        final int y = pixelIndex ~/ width;

        if (x < width - 1) {
          final neighborIdx = i + 4;
          final nr = data[neighborIdx].toDouble();
          final ng = data[neighborIdx + 1].toDouble();
          final nb = data[neighborIdx + 2].toDouble();
          final neighborBrightness = computeBrightness(nr, ng, nb);
          textureAccumulator += (brightness - neighborBrightness).abs() / 255.0;
          neighborSamples++;
        }
        if (y < height - 1) {
          final neighborIdx = i + (width * 4);
          if (neighborIdx < data.length) {
            final nr = data[neighborIdx].toDouble();
            final ng = data[neighborIdx + 1].toDouble();
            final nb = data[neighborIdx + 2].toDouble();
            final neighborBrightness = computeBrightness(nr, ng, nb);
            textureAccumulator += (brightness - neighborBrightness).abs() / 255.0;
            neighborSamples++;
          }
        }
      }

      if (total == 0) {
        return const _ImageHeuristics();
      }

      final double brownRatio = brownish / total;
      final double meanBrightness = brightnessSum / total;
      final double variance = (brightnessSqSum / total) - (meanBrightness * meanBrightness);
      final double contrast = variance <= 0 ? 0.0 : math.sqrt(variance);
      final double textureScore =
          neighborSamples == 0 ? 0.0 : textureAccumulator / neighborSamples;

      return _ImageHeuristics(
        brownRatio: brownRatio.clamp(0.0, 1.0).toDouble(),
        textureScore: textureScore.clamp(0.0, 1.0).toDouble(),
        contrast: contrast,
      );
    } catch (_) {
      return const _ImageHeuristics();
    }
  }

  double _computeVisualBeanScore({
    required double brownRatio,
    required double textureScore,
    required double contrastScore,
  }) {
    double normalize(double value, double maxValue) {
      if (maxValue <= 0) return 0.0;
      return math.min(1.0, math.max(0.0, value / maxValue));
    }

    final double brownComponent = normalize(brownRatio, 0.12); // ~12% brown pixels for solid beans
    final double textureComponent = normalize(textureScore, 0.16); // texture differences across neighbors
    final double contrastComponent = normalize(contrastScore, 18.0); // contrast across lighting

    return brownComponent * 0.5 + textureComponent * 0.3 + contrastComponent * 0.2;
  }

  int _indexOfMax(List<double> values) {
    if (values.isEmpty) {
      return -1;
    }
    int bestIndex = 0;
    double bestValue = values[0];
    for (int i = 1; i < values.length; i++) {
      final double value = values[i];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  @override
  Widget build(BuildContext context) {
    if (!_isPermissionGranted) {
      return _buildPermissionRequest();
    }

    return Stack(
      children: [
        Container(
          color: AppColors.scanDarkGrey,
          child: SafeArea(
            child: Column(
              children: [
                _buildHeader(context),
                _buildTitleAndInstructions(),
                const SizedBox(height: 12), // Balanced spacing
                _buildCameraViewfinder(),
                const SizedBox(height: 12), // Balanced spacing
                _buildUploadButton(),
                const SizedBox(height: 8), // Smaller spacing before controls
                _buildCameraControls(),
              ],
            ),
          ),
        ),
        Positioned(
          top: 0,
          left: 0,
          right: 0,
          child: _buildOverrideBanner(),
        ),
      ],
    );
  }

  Widget _buildPermissionRequest() {
    return Container(
      color: AppColors.scanDarkGrey,
      child: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.camera_alt,
                size: 80,
                color: Colors.white,
              ),
              const SizedBox(height: 20),
              const Text(
                'Camera Permission Required',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              const Text(
                'This app needs camera access to scan coffee beans.',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.white70,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () async {
                  // Try a more direct approach
                  debugPrint('User tapped Grant Permission button');
                  
                  // Force a direct permission request with proper Android registration
                  try {
                    debugPrint('=== Direct permission request ===');
                    
                    // Request camera permission first
                    debugPrint('Requesting camera permission directly...');
                    final cameraResult = await Permission.camera.request();
                    debugPrint('Direct camera permission result: $cameraResult');
                    
                    // Wait for camera permission to be processed
                    await Future.delayed(const Duration(milliseconds: 1500));
                    
                    // Request storage permission
                    debugPrint('Requesting storage permission directly...');
                    final storageResult = await Permission.storage.request();
                    debugPrint('Direct storage permission result: $storageResult');
                    
                    // Wait for storage permission to be processed
                    await Future.delayed(const Duration(milliseconds: 1500));
                    
                    // Now check permissions again
                    debugPrint('Checking final permission status...');
                    await _checkPermissions();
                  } catch (e) {
                    debugPrint('Direct permission request error: $e');
                    await _checkPermissions();
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primaryBrown,
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                ),
                child: const Text(
                  'Grant Permission',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              TextButton(
                onPressed: _openAppSettings,
                style: TextButton.styleFrom(
                  foregroundColor: Colors.white70,
                ),
                child: const Text(
                  'Open App Settings',
                  style: TextStyle(
                    fontSize: 14,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(
        left: AppConstants.largePadding,
        right: AppConstants.largePadding,
        top: 12,
        bottom: 8,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Container(
            decoration: BoxDecoration(
              color: Colors.black26,
              borderRadius: BorderRadius.circular(20),
            ),
            child: IconButton(
              icon: const Icon(
                Icons.close,
                color: Colors.white,
                size: 20,
              ),
              onPressed: () {
                if (widget.onClose != null) {
                  widget.onClose!();
                } else {
                  if (Navigator.of(context).canPop()) {
                    Navigator.of(context).pop();
                  }
                }
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTitleAndInstructions() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
      child: Column(
        children: [
          const Text(
            "Bean Scanner & Defect Detector",
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w600,
              color: Colors.white,
              letterSpacing: 0.5,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 6),
          const Text(
            "Point your camera at coffee beans to identify their type and detect defects (insect damage, quaker, shell, etc.).",
            style: TextStyle(
              fontSize: 13,
              color: Colors.white70,
              height: 1.4,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
        ],
      ),
    );
  }

  Widget _buildCameraViewfinder() {
    return Expanded(
      flex: 4, // Increased from 3 to 4 for more height
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (_isCameraInitialized && _cameraController != null && _cameraController!.value.isInitialized)
            _buildFullCameraPreview()
          else
            _buildCameraPlaceholder(),
          if (_isCameraInitialized && _cameraController != null && _cameraController!.value.isInitialized)
            _buildCornerBrackets(),
          // Brightness Control - Fixed vertical slider on the right side (like phone camera)
          if (_isCameraInitialized && _cameraController != null && _cameraController!.value.isInitialized && _showBrightnessControl)
            Positioned(
              right: 20,
              top: 0,
              bottom: 0,
              child: AnimatedOpacity(
                opacity: _showBrightnessControl ? 1.0 : 0.0,
                duration: const Duration(milliseconds: 200),
                child: Center(
                  child: GestureDetector(
                    behavior: HitTestBehavior.opaque, // Make sure gestures are captured
                    onVerticalDragStart: (details) {
                      // Store the starting position for relative movement
                    },
                    onVerticalDragUpdate: (details) async {
                      if (_cameraController == null || !_cameraController!.value.isInitialized) return;
                      
                      // Calculate new brightness based on drag position
                      // The slider is 200px tall, so map the drag to that range
                      final sliderHeight = 200.0;
                      final totalRange = _maxExposure - _minExposure;
                      final sensitivity = totalRange / sliderHeight;
                      final delta = -details.primaryDelta! * sensitivity;
                      final newExposure = (_currentExposure + delta).clamp(_minExposure, _maxExposure);
                      
                      _currentExposure = newExposure;
                      try { 
                        await _cameraController!.setExposureOffset(_currentExposure); 
                      } catch (e) {
                        debugPrint('Exposure error: $e');
                      }
                      if (mounted) setState(() {});
                    },
                    onTapDown: (details) async {
                      if (_cameraController == null || !_cameraController!.value.isInitialized) return;
                      
                      // Calculate brightness based on tap position
                      final sliderHeight = 200.0;
                      final localY = details.localPosition.dy;
                      final normalizedY = (localY / sliderHeight).clamp(0.0, 1.0);
                      // Top = max exposure (bright), bottom = min exposure (dark)
                      final newExposure = _maxExposure - (normalizedY * (_maxExposure - _minExposure));
                      
                      _currentExposure = newExposure.clamp(_minExposure, _maxExposure);
                      try { 
                        await _cameraController!.setExposureOffset(_currentExposure); 
                      } catch (e) {
                        debugPrint('Exposure error: $e');
                      }
                      if (mounted) setState(() {});
                    },
                    child: Container(
                      width: 60, // Wider touch area for easier interaction
                      height: 200, // Shorter slider
                      padding: const EdgeInsets.symmetric(horizontal: 18),
                      child: Stack(
                        children: [
                          // Vertical line - thin orange line through center
                          Positioned(
                            left: 18,
                            top: 0,
                            bottom: 0,
                            child: Container(
                              width: 2, // Thin line
                              decoration: BoxDecoration(
                                color: Colors.orange,
                                borderRadius: BorderRadius.circular(1),
                              ),
                            ),
                          ),
                          // Sun icon with 8 rays - positioned on the vertical line
                          Positioned(
                            left: 1, // Center on the line (line center is at x=19, sun is 36px wide, so left: 19-18=1)
                            top: (_maxExposure - _currentExposure) / (_maxExposure - _minExposure) * 200 - 18,
                            child: CustomPaint(
                              size: const Size(36, 36),
                              painter: _SunIconPainter(),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildCameraPlaceholder() {
    return Container(
      width: double.infinity,
      height: double.infinity,
      decoration: BoxDecoration(
        color: Colors.grey[200],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.camera_alt,
            size: AppConstants.extraLargeIconSize,
            color: Colors.grey,
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          if (_cameraController != null && !_isCameraInitialized)
            const CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(AppColors.primaryBrown),
            )
          else
            const Text(
              'Initializing camera...',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 14,
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildFullCameraPreview() {
    return GestureDetector(
      // Handle pinch zoom (two or more fingers) - works in all directions
      onScaleStart: (details) {
        // Only handle zoom if 2+ fingers
        if (details.pointerCount >= 2) {
          _gestureBaseZoom = _currentZoom;
        }
      },
      onScaleUpdate: (details) async {
        if (_cameraController == null || !_cameraController!.value.isInitialized) return;
        // Only handle zoom if it's a pinch gesture (2+ fingers)
        // Zoom works in all directions - scale is direction-independent
        if (details.pointerCount >= 2) {
          // Calculate zoom based on scale - works regardless of pinch direction
          // Scale is calculated from the distance between two pointers, so it works in all directions
          final desired = (_gestureBaseZoom * details.scale).clamp(_minZoom, _maxZoom);
          _currentZoom = desired;
          try { 
            await _cameraController!.setZoomLevel(_currentZoom); 
          } catch (e) {
            debugPrint('Zoom error: $e');
          }
          if (mounted) setState(() {});
        }
      },
      onScaleEnd: (details) {
        // Zoom gesture ended
      },
      // Handle tap to cycle demo override (or toggle brightness when override is off)
      onTap: _handleCameraTap,
      onLongPress: _toggleBrightnessControlVisibility,
      child: CameraPreview(_cameraController!),
    );
  }

  Widget _buildCornerBrackets() {
    return Stack(
      children: [
        // Top-left corner
        Positioned(
          top: 10,
          left: 10,
          child: Container(
            width: AppConstants.iconButtonSize,
            height: AppConstants.iconButtonSize,
            decoration: const BoxDecoration(
              border: Border(
                top: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
                left: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
              ),
            ),
          ),
        ),
        // Top-right corner
        Positioned(
          top: 10,
          right: 10,
          child: Container(
            width: AppConstants.iconButtonSize,
            height: AppConstants.iconButtonSize,
            decoration: const BoxDecoration(
              border: Border(
                top: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
                right: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
              ),
            ),
          ),
        ),
        // Bottom-left corner
        Positioned(
          bottom: 10,
          left: 10,
          child: Container(
            width: AppConstants.iconButtonSize,
            height: AppConstants.iconButtonSize,
            decoration: const BoxDecoration(
              border: Border(
                bottom: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
                left: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
              ),
            ),
          ),
        ),
        // Bottom-right corner
        Positioned(
          bottom: 10,
          right: 10,
          child: Container(
            width: AppConstants.iconButtonSize,
            height: AppConstants.iconButtonSize,
            decoration: const BoxDecoration(
              border: Border(
                bottom: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
                right: BorderSide(color: AppColors.primaryBrown, width: AppConstants.thickBorder),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildOverrideBanner() {
    final demo = DemoOverride();
    if (!demo.isEnabled) {
      return const SizedBox.shrink();
    }

    final String? overrideType = demo.overrideBeanType;
    final _OverrideShapeOption activeOption = (() {
      if (overrideType == null) return _overrideShapeOrder.first;
      return _overrideShapeOrder.firstWhere(
        (option) => option.beanType.toLowerCase() == overrideType.toLowerCase(),
        orElse: () => _overrideShapeOrder.first,
      );
    })();

    return SafeArea(
      bottom: false,
      child: Align(
        alignment: Alignment.topLeft,
        child: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTap: _cycleOverrideSelection,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
            child: Icon(
              activeOption.icon,
              color: Colors.white.withOpacity(0.1),
              size: 22,
            ),
          ),
        ),
      ),
    );
  }


  Widget _buildUploadButton() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
      child: GestureDetector(
        onTap: _pickImageFromGallery,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.1),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.photo_library_outlined,
                color: AppColors.textDarkGrey,
                size: 18,
              ),
              const SizedBox(width: 8),
              const Text(
                "Upload From Gallery",
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: AppColors.textDarkGrey,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCameraControls() {
    return Padding(
      padding: const EdgeInsets.only(
        left: AppConstants.largePadding,
        right: AppConstants.largePadding,
        top: 12,
        bottom: 16,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          // Flash control
          Container(
            decoration: BoxDecoration(
              color: Colors.black26,
              borderRadius: BorderRadius.circular(25),
            ),
            child: IconButton(
              icon: Icon(
                _isFlashOn ? Icons.flash_on : Icons.flash_off,
                color: _isFlashOn ? Colors.amber : Colors.white,
                size: 24,
              ),
              onPressed: _isCameraInitialized ? _toggleFlash : null,
            ),
          ),
          
          // Shutter button
          GestureDetector(
            onTap: _isCameraInitialized ? _takePicture : null,
            child: Container(
              width: 70,
              height: 70,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.2),
                border: Border.all(
                  color: Colors.white,
                  width: 3,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.3),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: const Center(
                child: Icon(
                  Icons.camera_alt,
                  color: Colors.white,
                  size: 32,
                ),
              ),
            ),
          ),
          
          // Camera switch
          Container(
            decoration: BoxDecoration(
              color: Colors.black26,
              borderRadius: BorderRadius.circular(25),
            ),
            child: IconButton(
              icon: const Icon(
                Icons.flip_camera_ios,
                color: Colors.white,
                size: 24,
              ),
              onPressed: _isCameraInitialized && _cameras.length > 1 ? _switchCamera : null,
            ),
          ),
        ],
      ),
    );
  }
}

class _ImageHeuristics {
  final double brownRatio;
  final double textureScore;
  final double contrast;

  const _ImageHeuristics({
    this.brownRatio = 0.0,
    this.textureScore = 0.0,
    this.contrast = 0.0,
  });
}
