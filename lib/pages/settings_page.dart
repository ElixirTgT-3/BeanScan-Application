import 'package:flutter/material.dart';
import '../utils/app_constants.dart';
import '../utils/app_settings.dart';
import 'help_center_page.dart';
import 'about_page.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  final AppSettings _settings = AppSettings.instance;

  static const Color _accentTextColor = Color(0xFF554848);
  static const Color _surfaceGrey = Color(0xFFF0F0F0);
  static const Color _cardBorderColor = Color(0xFFDEDEDE);
  static const Color _iconBackgroundColor = Color(0xFFE0D3B8);
  static const Color _cardBackgroundBase = Color(0xFFFFE6B2);
  static const Color _darkSurface = Color(0xFF212121);
  static const Color _darkCardBorder = Color(0xFF3B3329);
  static const Color _darkCardBackground = Color(0xFF2A231A);
  static const Color _darkHeaderText = Color(0xFFE5C69F);
  static const Color _darkBodyText = Color(0xFFF2E0C4);
  static const Color _darkIconBackground = Color(0xFF463422);
  static const Color _darkIconColor = Color(0xFFFFD7A4);

  @override
  void initState() {
    super.initState();
    _settings.addListener(_handleSettingsChanged);
  }

  @override
  void dispose() {
    _settings.removeListener(_handleSettingsChanged);
    super.dispose();
  }

  void _handleSettingsChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final bool isDark = colorScheme.brightness == Brightness.dark;
    final Color background = isDark ? _darkSurface : _surfaceGrey;
    return ColoredBox(
      color: background,
      child: Column(
        children: [
          _buildHeader(colorScheme, background),
          _buildContent(colorScheme, background),
        ],
      ),
    );
  }

  Widget _buildHeader(ColorScheme colorScheme, Color backgroundColor) {
    final bool isDark = colorScheme.brightness == Brightness.dark;
    final Color titleColor = isDark ? _darkHeaderText : _accentTextColor;
    return Container(
      width: double.infinity,
      color: backgroundColor,
      child: SafeArea(
        bottom: false,
        child: Container(
          padding: const EdgeInsets.symmetric(
            vertical: AppConstants.headerPadding,
            horizontal: AppConstants.largePadding,
          ),
          child: Text(
            "Settings",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: titleColor,
            ),
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }

  Widget _buildContent(ColorScheme colorScheme, Color background) {
    final bool isDark = colorScheme.brightness == Brightness.dark;
    return Expanded(
      child: Container(
        width: double.infinity,
        color: background,
        child: ListView(
          physics: const ClampingScrollPhysics(),
          padding: const EdgeInsets.symmetric(
            horizontal: AppConstants.largePadding,
            vertical: AppConstants.largePadding,
          ),
          children: [
            _buildSection(
              colorScheme: colorScheme,
              title: "General Preferences",
              children: [
                _buildSettingItem(
                  icon: Icons.download,
                  title: "Auto-Save Scans",
                  colorScheme: colorScheme,
                  trailing: Switch(
                    value: _settings.autoSaveScans,
                    thumbColor: WidgetStateProperty.resolveWith(
                      (states) => states.contains(WidgetState.selected)
                          ? (isDark ? _darkHeaderText : _accentTextColor)
                          : null,
                    ),
                    activeTrackColor:
                        (isDark ? _darkHeaderText : _accentTextColor).withValues(alpha: 0.35),
                    inactiveTrackColor:
                        (isDark ? _darkIconBackground : _iconBackgroundColor).withValues(alpha: isDark ? 0.6 : 0.6),
                    onChanged: (value) => _settings.setAutoSaveScans(value),
                  ),
                  onTap: () => _settings.setAutoSaveScans(!_settings.autoSaveScans),
                ),
                _buildDivider(colorScheme),
                _buildSettingItem(
                  icon: Icons.dark_mode,
                  title: "Theme",
                  colorScheme: colorScheme,
                  subtitle: _describeThemeMode(_settings.themeMode),
                  onTap: () => _showThemePicker(context),
                ),
              ],
            ),
            _buildSection(
              colorScheme: colorScheme,
              title: "Support & About",
              children: [
                _buildSettingItem(
                  icon: Icons.help_outline,
                  title: "Help Center",
                  colorScheme: colorScheme,
                  onTap: () => _openHelpCenter(context),
                ),
                _buildDivider(colorScheme),
                _buildSettingItem(
                  icon: Icons.info_outline,
                  title: "About the App",
                  colorScheme: colorScheme,
                  onTap: () => _openAbout(context),
                ),
              ],
            ),
            const SizedBox(height: AppConstants.largeSpacing),
          ],
        ),
      ),
    );
  }

  Widget _buildSection({
    required String title,
    required List<Widget> children,
    required ColorScheme colorScheme,
  }) {
    final bool isDark = colorScheme.brightness == Brightness.dark;
    final Color cardBackground =
        isDark ? _darkCardBackground : _cardBackgroundBase.withValues(alpha: 0.45);
    final Color borderColor = isDark ? _darkCardBorder : _cardBorderColor;
    final Color titleColor = isDark ? _darkBodyText : _accentTextColor.withValues(alpha: 1.0);
    return Container(
      margin: const EdgeInsets.only(bottom: AppConstants.largePadding),
      padding: const EdgeInsets.symmetric(
        horizontal: AppConstants.largePadding,
        vertical: AppConstants.largePadding,
      ),
      decoration: BoxDecoration(
        color: cardBackground,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: borderColor.withValues(alpha: isDark ? 1.0 : 1.0),
          width: AppConstants.thinBorder,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: AppConstants.mediumSpacing),
            child: Text(
              title,
              style: TextStyle(
                color: titleColor,
                fontSize: 15,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          ...children,
        ],
      ),
    );
  }

  Widget _buildSettingItem({
    required IconData icon,
    required String title,
    required ColorScheme colorScheme,
    VoidCallback? onTap,
    Widget? trailing,
    String? subtitle,
  }) {
    final bool isDark = colorScheme.brightness == Brightness.dark;
    final Color baseTextColor = isDark ? _darkBodyText : _accentTextColor;
    final Color subtitleColor =
        isDark ? _darkBodyText.withValues(alpha: 0.78) : baseTextColor.withValues(alpha: 0.7);
    final Color trailingColor =
        isDark ? _darkIconColor.withValues(alpha: 0.85) : baseTextColor.withValues(alpha: 0.6);
    final Color iconBackground = isDark ? _darkIconBackground : _iconBackgroundColor;

    final Widget content = Padding(
      padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: iconBackground,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(
              icon,
              color: isDark ? _darkIconColor : _accentTextColor,
              size: 22,
            ),
          ),
          const SizedBox(width: AppConstants.mediumSpacing),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: baseTextColor,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (subtitle != null)
                  Text(
                    subtitle,
                    style: TextStyle(
                      color: subtitleColor,
                      fontSize: 12,
                    ),
                  ),
              ],
            ),
          ),
          trailing ??
              Icon(
                Icons.chevron_right,
                color: trailingColor,
                size: 20,
              ),
        ],
      ),
    );

    if (onTap == null && trailing == null) {
      return content;
    }

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: content,
      ),
    );
  }

  Widget _buildDivider(ColorScheme colorScheme) {
    final bool isDark = colorScheme.brightness == Brightness.dark;
    final Color baseColor = isDark ? _darkCardBorder : _cardBorderColor;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.smallSpacing),
      child: Divider(
        color: baseColor.withValues(
          alpha: isDark ? 0.45 : 0.6,
        ),
        thickness: AppConstants.thinBorder,
        height: 1,
      ),
    );
  }

  String _describeThemeMode(ThemeMode mode) {
    switch (mode) {
      case ThemeMode.light:
        return "Light mode";
      case ThemeMode.dark:
        return "Dark mode";
      default:
        return "Light mode";
    }
  }

  Future<void> _showThemePicker(BuildContext context) async {
    final colorScheme = Theme.of(context).colorScheme;
    final themeMode = await showModalBottomSheet<ThemeMode>(
      context: context,
      backgroundColor: colorScheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 16),
            Text(
              'Choose Theme',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: colorScheme.primary,
              ),
            ),
            const SizedBox(height: 8),
            _buildThemeOption(context, colorScheme, ThemeMode.light, 'Light mode'),
            _buildThemeOption(context, colorScheme, ThemeMode.dark, 'Dark mode'),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );

    if (themeMode != null) {
      await _settings.setThemeMode(themeMode);
    }
  }

  Widget _buildThemeOption(BuildContext context, ColorScheme colorScheme, ThemeMode mode, String label) {
    final isSelected = _settings.themeMode == mode;
    return ListTile(
      onTap: () => Navigator.of(context).pop(mode),
      leading: Icon(
        mode == ThemeMode.dark
            ? Icons.nights_stay
            : mode == ThemeMode.light
                ? Icons.wb_sunny
                : Icons.settings_suggest,
        color: colorScheme.primary,
      ),
      title: Text(
        label,
        style: TextStyle(color: colorScheme.onSurface),
      ),
      trailing: isSelected
          ? Icon(Icons.check, color: colorScheme.primary)
          : null,
    );
  }

  void _openHelpCenter(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const HelpCenterPage()),
    );
  }

  void _openAbout(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const AboutPage()),
    );
  }
}
