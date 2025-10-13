import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
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
    return Column(
      children: [
        _buildHeader(),
        _buildContent(),
      ],
    );
  }

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: AppColors.headerGrey,
      ),
      child: SafeArea(
        bottom: false,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: AppConstants.headerPadding),
          child: const Text(
            "Settings",
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: AppColors.primaryBrown,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContent() {
    return Expanded(
      child: Container(
        color: AppColors.headerGrey,
        child: ListView(
          padding: const EdgeInsets.all(AppConstants.defaultPadding),
          children: [
            _buildSection(
              title: "General Preferences",
              children: [
                _buildSettingItem(
                  icon: Icons.download,
                  title: "Auto-Save Scans",
                  trailing: Switch(
                    value: _settings.autoSaveScans,
                    activeColor: AppColors.primaryBrown,
                    onChanged: (value) => _settings.setAutoSaveScans(value),
                  ),
                  onTap: () => _settings.setAutoSaveScans(!_settings.autoSaveScans),
                ),
                _buildDivider(),
                _buildSettingItem(
                  icon: Icons.dark_mode,
                  title: "Theme",
                  subtitle: _describeThemeMode(_settings.themeMode),
                  onTap: () => _showThemePicker(context),
                ),
              ],
            ),
            _buildSection(
              title: "Support & About",
              children: [
                _buildSettingItem(
                  icon: Icons.help_outline,
                  title: "Help Center",
                  onTap: () => _openHelpCenter(context),
                ),
                _buildDivider(),
                _buildSettingItem(
                  icon: Icons.info_outline,
                  title: "About the App",
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

  Widget _buildSection({required String title, required List<Widget> children}) {
    return Container(
      margin: const EdgeInsets.only(bottom: AppConstants.largeSpacing),
      decoration: BoxDecoration(
        color: AppColors.lightBeige,
        borderRadius: BorderRadius.circular(AppConstants.largeRadius),
        border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppConstants.largePadding,
              AppConstants.largePadding,
              AppConstants.largePadding,
              AppConstants.smallSpacing,
            ),
            child: Text(
              title,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: AppColors.primaryBrown,
              ),
            ),
          ),
          ...children,
          const SizedBox(height: AppConstants.smallSpacing),
        ],
      ),
    );
  }

  Widget _buildSettingItem({
    required IconData icon,
    required String title,
    VoidCallback? onTap,
    Widget? trailing,
    String? subtitle,
  }) {
    final content = Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: AppConstants.largePadding,
        vertical: AppConstants.smallSpacing,
      ),
      child: Row(
        children: [
          Container(
            width: AppConstants.iconButtonSize,
            height: AppConstants.iconButtonSize,
            decoration: BoxDecoration(
              color: AppColors.iconBackground,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.primaryBrown.withOpacity(0.2), width: AppConstants.thinBorder),
            ),
            child: Icon(
              icon,
              color: AppColors.primaryBrown,
              size: AppConstants.smallIconSize,
            ),
          ),
          const SizedBox(width: AppConstants.mediumSpacing),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: AppColors.primaryBrown,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (subtitle != null)
                  Text(
                    subtitle,
                    style: const TextStyle(
                      color: AppColors.textGrey,
                      fontSize: 12,
                    ),
                  ),
              ],
            ),
          ),
          trailing ??
              const Icon(
                Icons.chevron_right,
                color: AppColors.primaryBrown,
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

  Widget _buildDivider() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
      child: const Divider(
        color: AppColors.dividerGrey,
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
    final themeMode = await showModalBottomSheet<ThemeMode>(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 16),
            const Text(
              'Choose Theme',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: AppColors.primaryBrown,
              ),
            ),
            const SizedBox(height: 8),
            _buildThemeOption(context, ThemeMode.light, 'Light mode'),
            _buildThemeOption(context, ThemeMode.dark, 'Dark mode'),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );

    if (themeMode != null) {
      await _settings.setThemeMode(themeMode);
    }
  }

  Widget _buildThemeOption(BuildContext context, ThemeMode mode, String label) {
    final isSelected = _settings.themeMode == mode;
    return ListTile(
      onTap: () => Navigator.of(context).pop(mode),
      leading: Icon(
        mode == ThemeMode.dark
            ? Icons.nights_stay
            : mode == ThemeMode.light
                ? Icons.wb_sunny
                : Icons.settings_suggest,
        color: AppColors.primaryBrown,
      ),
      title: Text(label),
      trailing: isSelected
          ? const Icon(Icons.check, color: AppColors.primaryBrown)
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
