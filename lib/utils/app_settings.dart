import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AppSettings extends ChangeNotifier {
  AppSettings._internal();

  static final AppSettings instance = AppSettings._internal();

  static const _autoSaveKey = 'auto_save_scans';
  static const _themeModeKey = 'theme_mode';
  bool _autoSaveScans = true;
  ThemeMode _themeMode = ThemeMode.light;
  SharedPreferences? _prefs;
  bool _isLoaded = false;

  bool get autoSaveScans => _autoSaveScans;
  ThemeMode get themeMode => _themeMode;
  bool get isLoaded => _isLoaded;

  Future<void> load() async {
    if (_isLoaded) return;
    _prefs = await SharedPreferences.getInstance();
    _autoSaveScans = _prefs?.getBool(_autoSaveKey) ?? true;
    _themeMode = _themeModeFromString(_prefs?.getString(_themeModeKey));
    _isLoaded = true;
    notifyListeners();
  }

  Future<void> setAutoSaveScans(bool value) async {
    _autoSaveScans = value;
    notifyListeners();
    await _prefs?.setBool(_autoSaveKey, value);
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    if (mode != ThemeMode.light && mode != ThemeMode.dark) {
      mode = ThemeMode.light;
    }
    if (_themeMode == mode) return;
    _themeMode = mode;
    notifyListeners();
    await _prefs?.setString(_themeModeKey, mode.name);
  }

  ThemeMode _themeModeFromString(String? value) {
    if (value == 'dark') return ThemeMode.dark;
    return ThemeMode.light;
  }
}
