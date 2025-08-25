# BeanScan Application

A Flutter application for scanning coffee beans to detect molds and defects.

## Project Structure

The project has been organized into a clean, maintainable structure:

```
lib/
├── main.dart                 # Main app entry point and navigation
├── pages/                    # Individual page components
│   ├── history_page.dart    # History page with empty state
│   ├── scan_page.dart       # Camera interface for scanning
│   └── settings_page.dart   # App settings and preferences
└── utils/                    # Utility classes and constants
    ├── app_colors.dart      # Centralized color definitions
    └── app_constants.dart   # App-wide constants and values
```

## Features

### 🏠 History Page
- Clean header with safe area support
- Empty state message for new users
- Consistent styling with the app theme

### 📷 Scan Page
- **Real Camera Integration** - Live camera preview with high-quality resolution
- **Professional Camera Controls** - Flash toggle, shutter button, camera switching
- **Permission Handling** - Automatic camera and storage permission requests
- **Image Capture** - Take photos and save them to device storage
- **Upload from Gallery** - Select existing images from device gallery
- **Camera Viewfinder** - Professional scanning interface with corner brackets
- **Dark Theme** - Optimized for camera usage and low-light conditions

### ⚙️ Settings Page
- Organized into logical sections:
  - Scanning Preferences
  - App Preferences  
  - Support & About
- Clean list interface with icons
- Consistent styling throughout

### 🧭 Navigation
- Bottom navigation bar (hidden on scan page)
- Center scan button for easy access
- Smooth transitions between pages

## Architecture Benefits

✅ **Separation of Concerns**: Each page is in its own file
✅ **Maintainability**: Easy to find and modify specific features
✅ **Reusability**: Colors and constants are centralized
✅ **Consistency**: All pages use the same design system
✅ **Scalability**: Easy to add new pages and features

## Color Scheme

- **Primary Brown**: `#55351C` - Main brand color
- **Light Beige**: `#F8F5ED` - Main background
- **Header Grey**: `#EEEEEE` - Page headers
- **Dark Grey**: `#333333` - Scan page background
- **Icon Background**: `#E0D8C7` - Settings icon backgrounds

## Getting Started

1. Ensure Flutter is installed and configured
2. Clone the repository
3. Run `flutter pub get` to install dependencies
4. Run `flutter run` to start the app

## Dependencies

- Flutter SDK
- Material Design components
- **camera** - Real-time camera functionality and image capture
- **permission_handler** - Camera and storage permission management

## Future Enhancements

- Real camera integration
- Image processing for coffee bean analysis
- Results storage and history
- User authentication
- Cloud sync capabilities
