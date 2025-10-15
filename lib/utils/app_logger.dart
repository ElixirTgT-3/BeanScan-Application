import 'dart:developer' as developer;

import 'package:flutter/foundation.dart';

/// Lightweight logger that keeps `print` statements out of production builds.
void logDebug(
  String category,
  String message, {
  Object? error,
  StackTrace? stackTrace,
}) {
  if (kDebugMode) {
    developer.log(
      message,
      name: category,
      error: error,
      stackTrace: stackTrace,
    );
  }
}
