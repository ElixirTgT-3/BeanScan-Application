import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';
import 'results_page.dart';

class ScanPage extends StatefulWidget {
  const ScanPage({super.key});

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

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Don't check permissions automatically - let user trigger it
    debugPrint('ScanPage initialized - waiting for user to request permissions');
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
      _checkPermissions();
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

    try {
      final XFile image = await _cameraController!.takePicture();
      debugPrint('Picture taken: ${image.path}');
      
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
      await _processImage(File(image.path));
      
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
    }
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
      await _cameraController!.setFlashMode(
        _isFlashOn ? FlashMode.off : FlashMode.torch,
      );
      setState(() {
        _isFlashOn = !_isFlashOn;
      });
    } catch (e) {
      debugPrint('Error toggling flash: $e');
    }
  }

  Future<void> _processImage(File imageFile) async {
    try {
      // Show loading dialog
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          return const AlertDialog(
            content: Row(
              children: [
                CircularProgressIndicator(),
                SizedBox(width: 20),
                Text("Analyzing bean type..."),
              ],
            ),
          );
        },
      );

      // Make API call
      final result = await ApiService.predictBeanTypeWithErrorHandling(imageFile);
      
      // Hide loading dialog
      if (mounted) {
        Navigator.of(context).pop();
      }

      if (result['success'] && result['prediction'] != null) {
        // Navigate to results page
        if (mounted) {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => ResultsPage(
                prediction: result['prediction'],
                imagePath: imageFile.path,
              ),
            ),
          );
        }
      } else {
        // Show error message
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(result['error'] ?? 'Failed to analyze image'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
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
        await _processImage(File(image.path));
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

  @override
  Widget build(BuildContext context) {
    if (!_isPermissionGranted) {
      return _buildPermissionRequest();
    }

    return Container(
      color: AppColors.scanDarkGrey,
      child: SafeArea(
        child: Column(
          children: [
            _buildHeader(context),
            _buildTitleAndInstructions(),
            const SizedBox(height: AppConstants.extraLargeSpacing),
            _buildCameraViewfinder(),
            const SizedBox(height: AppConstants.largeSpacing),
            _buildUploadButton(),
            const SizedBox(height: AppConstants.largeSpacing),
            _buildCameraControls(),
          ],
        ),
      ),
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
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding, vertical: AppConstants.mediumSpacing),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          IconButton(
            icon: const Icon(
              Icons.close,
              color: Colors.white,
              size: AppConstants.mediumIconSize,
            ),
            onPressed: () => Navigator.of(context).pop(),
          ),
        ],
      ),
    );
  }

  Widget _buildTitleAndInstructions() {
    return const Padding(
      padding: EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
      child: Column(
        children: [
          Text(
            "Bean Type Scanner",
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: AppConstants.smallSpacing),
          Text(
            "Point your camera at coffee beans to identify their type (Arabica, Robusta, Liberica, Excelsa).",
            style: TextStyle(
              fontSize: 14,
              color: Colors.white70,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildCameraViewfinder() {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: AppColors.primaryBrown,
              width: AppConstants.mediumBorder,
              style: BorderStyle.solid,
            ),
            borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
            child: Stack(
              children: [
                _isCameraInitialized && _cameraController != null && _cameraController!.value.isInitialized
                    ? CameraPreview(_cameraController!)
                    : _buildCameraPlaceholder(),
                if (_isCameraInitialized && _cameraController != null && _cameraController!.value.isInitialized)
                  _buildCornerBrackets(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCameraPlaceholder() {
    return Container(
      width: double.infinity,
      height: double.infinity,
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(AppConstants.smallRadius),
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


  Widget _buildUploadButton() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
      child: GestureDetector(
        onTap: _pickImageFromGallery,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing, horizontal: AppConstants.largePadding),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(AppConstants.extraLargeRadius),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Upload From Gallery",
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDarkGrey,
                ),
              ),
              const SizedBox(width: AppConstants.smallSpacing),
              Icon(
                Icons.upload,
                color: AppColors.textDarkGrey,
                size: AppConstants.smallIconSize,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCameraControls() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding, vertical: AppConstants.mediumSpacing),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          // Flash control
          IconButton(
            icon: Icon(
              _isFlashOn ? Icons.flash_on : Icons.flash_off,
              color: Colors.white,
              size: AppConstants.mediumIconSize,
            ),
            onPressed: _isCameraInitialized ? _toggleFlash : null,
          ),
          
          // Shutter button
          GestureDetector(
            onTap: _isCameraInitialized ? _takePicture : null,
            child: Container(
              width: AppConstants.shutterButtonSize,
              height: AppConstants.shutterButtonSize,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: Colors.white,
                  width: AppConstants.thickBorder,
                ),
              ),
              child: const Center(
                child: Icon(
                  Icons.camera,
                  color: Colors.white,
                  size: AppConstants.largeIconSize,
                ),
              ),
            ),
          ),
          
          // Camera switch
          IconButton(
            icon: const Icon(
              Icons.flip_camera_ios,
              color: Colors.white,
              size: AppConstants.mediumIconSize,
            ),
            onPressed: _isCameraInitialized && _cameras.length > 1 ? _switchCamera : null,
          ),
        ],
      ),
    );
  }
} 