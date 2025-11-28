// Demo Override Utility
//
// Allows secretly changing the predicted coffee bean class before results are shown.
// Toggle override from the About screen and tap the camera view to cycle bean shapes.
//
// How it works:
// 1. Enable override via the hidden triple-tap in Settings > About
// 2. Tap the camera view to cycle bean shapes (square=Excelsa, oval=Arabica, triangle=Liberica, circle=Robusta)
// 3. The selected override is applied to the prediction before showing results
// 4. If no taps are made, the original AI prediction is used

import 'package:flutter/material.dart';

/// The 4 coffee bean types in CoffeeNet order
const List<String> kBeanTypes = ['Liberica', 'Arabica', 'Robusta', 'Excelsa'];

/// Singleton to manage demo override state
class DemoOverride {
  static final DemoOverride _instance = DemoOverride._internal();
  factory DemoOverride() => _instance;
  DemoOverride._internal();

  /// The currently selected override index (-1 = no override, use AI prediction)
  int _overrideIndex = -1;
  
  /// Whether demo mode is enabled (triple-tap About to toggle)
  bool _enabled = false;
  
  /// Get whether demo mode is enabled
  bool get isEnabled => _enabled;
  
  /// Toggle demo mode on/off
  void toggle() {
    _enabled = !_enabled;
    if (!_enabled) {
      _overrideIndex = -1; // Reset override when disabling
    }
  }
  
  /// Enable demo mode
  void enable() {
    _enabled = true;
  }
  
  /// Disable demo mode
  void disable() {
    _enabled = false;
    _overrideIndex = -1;
  }
  
  /// Get the current override bean type (null if no override)
  String? get overrideBeanType {
    if (!_enabled || _overrideIndex < 0 || _overrideIndex >= kBeanTypes.length) {
      return null;
    }
    return kBeanTypes[_overrideIndex];
  }
  
  /// Get the current override index
  int get overrideIndex => _overrideIndex;
  
  /// Set override to a specific bean type
  void setOverride(String beanType) {
    final index = kBeanTypes.indexOf(beanType);
    if (index >= 0) {
      _overrideIndex = index;
    }
  }
  
  /// Cycle to next bean type
  void nextBeanType() {
    if (!_enabled) return;
    _overrideIndex = (_overrideIndex + 1) % kBeanTypes.length;
  }
  
  /// Cycle to previous bean type
  void previousBeanType() {
    if (!_enabled) return;
    if (_overrideIndex < 0) {
      _overrideIndex = kBeanTypes.length - 1;
      return;
    }
    _overrideIndex = (_overrideIndex - 1 + kBeanTypes.length) % kBeanTypes.length;
  }
  
  /// Reset the override (use AI prediction)
  void resetOverride() {
    _overrideIndex = -1;
  }
  
  /// Initialize override based on AI prediction
  void initFromPrediction(String aiPrediction) {
    if (!_enabled) return;
    final index = kBeanTypes.indexOf(aiPrediction);
    _overrideIndex = index >= 0 ? index : 0;
  }
  
  /// Apply override to a prediction, returning the modified prediction
  /// Returns the original if no override is set
  Map<String, dynamic> applyOverride(Map<String, dynamic> predictionData) {
    final overrideType = overrideBeanType;
    if (overrideType == null) {
      return predictionData;
    }

    final modified = Map<String, dynamic>.from(predictionData);

    const possibleKeys = [
      'prediction',
      'predicted_class',
      'class',
      'bean_type',
      'bean_type_name',
    ];
    for (final key in possibleKeys) {
      modified[key] = overrideType;
    }

    final int positiveHash = overrideType.hashCode.abs();
    final double overrideConfidence =
        0.85 + (0.10 * ((positiveHash % 100) / 100.0));
    modified['confidence'] = overrideConfidence;
    modified['confidence_score'] = overrideConfidence;
    modified['probability'] = overrideConfidence;

    void applyProbabilityOverride(String key) {
      final source = modified[key];
      if (source is List) {
        final newProbs = List<double>.filled(kBeanTypes.length, 0.0);
        final overrideIdx = kBeanTypes.indexOf(overrideType);
        for (int i = 0; i < newProbs.length; i++) {
          if (i == overrideIdx) {
            newProbs[i] = overrideConfidence;
          } else {
            newProbs[i] =
                (1.0 - overrideConfidence) / (kBeanTypes.length - 1);
          }
        }
        modified[key] = newProbs;
      } else if (source is Map) {
        final newProbs = <String, double>{};
        for (final type in kBeanTypes) {
          if (type == overrideType) {
            newProbs[type] = overrideConfidence;
          } else {
            newProbs[type] =
                (1.0 - overrideConfidence) / (kBeanTypes.length - 1);
          }
        }
        modified[key] = newProbs;
      }
    }

    applyProbabilityOverride('all_probabilities');
    applyProbabilityOverride('probabilities');

    return modified;
  }
}

/// A widget that shows the demo override loading screen (read-only indicator)
class DemoOverrideLoadingOverlay extends StatelessWidget {
  final Widget child;

  const DemoOverrideLoadingOverlay({
    super.key,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    final demo = DemoOverride();
    if (!demo.isEnabled) {
      return child;
    }

    // Demo is enabled; keep UI invisible while still applying overrides.
    return child;
  }
}

IconData _shapeIconForBean(String beanType) {
  switch (beanType.toLowerCase()) {
    case 'excelsa':
      return Icons.crop_square_rounded;
    case 'arabica':
      return Icons.egg_outlined;
    case 'liberica':
      return Icons.change_history;
    case 'robusta':
      return Icons.circle_outlined;
    default:
      return Icons.catching_pokemon;
  }
}

