import 'package:shared_preferences/shared_preferences.dart';

enum AppThemeMode { system, light, dark }

class SettingsService {
  static const String _keyAutoSave = 'auto_save_scans';
  static const String _keyThemeMode = 'theme_mode'; // system|light|dark

  static Future<SharedPreferences> _prefs() async =>
      await SharedPreferences.getInstance();

  // Auto-save
  static Future<bool> getAutoSave() async {
    final prefs = await _prefs();
    return prefs.getBool(_keyAutoSave) ?? true; // default ON
  }

  static Future<void> setAutoSave(bool value) async {
    final prefs = await _prefs();
    await prefs.setBool(_keyAutoSave, value);
  }

  // Theme
  static Future<AppThemeMode> getThemeMode() async {
    final prefs = await _prefs();
    final stored = prefs.getString(_keyThemeMode) ?? 'system';
    switch (stored) {
      case 'light':
        return AppThemeMode.light;
      case 'dark':
        return AppThemeMode.dark;
      default:
        return AppThemeMode.system;
    }
  }

  static Future<void> setThemeMode(AppThemeMode mode) async {
    final prefs = await _prefs();
    final value = switch (mode) {
      AppThemeMode.system => 'system',
      AppThemeMode.light => 'light',
      AppThemeMode.dark => 'dark',
    };
    await prefs.setString(_keyThemeMode, value);
  }
}


