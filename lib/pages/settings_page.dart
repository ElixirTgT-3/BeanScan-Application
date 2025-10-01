import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/settings_service.dart';
import 'package:url_launcher/url_launcher.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  bool _autoSave = true;
  AppThemeMode _theme = AppThemeMode.system;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final saved = await SettingsService.getAutoSave();
    final mode = await SettingsService.getThemeMode();
    if (!mounted) return;
    setState(() {
      _autoSave = saved;
      _theme = mode;
    });
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
                _buildToggleItem(Icons.download, "Auto-Save Scans", _autoSave, (v) async {
                  setState(() => _autoSave = v);
                  await SettingsService.setAutoSave(v);
                }),
                _buildDivider(),
                _buildSettingItem(Icons.dark_mode, "Theme", onTap: _showThemePicker,
                    trailing: Text(_theme == AppThemeMode.system ? 'System' : _theme == AppThemeMode.light ? 'Light' : 'Dark',
                      style: const TextStyle(color: AppColors.primaryBrown)))
              ],
            ),
            _buildSection(
              title: "Support & About",
              children: [
                _buildSettingItem(Icons.help_outline, "Help Center", onTap: _openHelpCenter),
                _buildDivider(),
                _buildSettingItem(Icons.info_outline, "About the App", onTap: _showAbout),
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

  Widget _buildSettingItem(IconData icon, String title, {VoidCallback? onTap, Widget? trailing}) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: AppConstants.largePadding,
        vertical: AppConstants.smallSpacing,
      ),
      child: InkWell(
        onTap: onTap,
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
            child: Text(
              title,
              style: const TextStyle(
                color: AppColors.primaryBrown,
                fontSize: 16,
              ),
            ),
          ),
          if (trailing != null) trailing,
          const Icon(Icons.chevron_right, color: AppColors.primaryBrown),
        ],
      ),
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

  Widget _buildToggleItem(IconData icon, String title, bool value, ValueChanged<bool> onChanged) {
    return Padding(
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
            child: Icon(icon, color: AppColors.primaryBrown, size: AppConstants.smallIconSize),
          ),
          const SizedBox(width: AppConstants.mediumSpacing),
          Expanded(
            child: Text(title, style: const TextStyle(color: AppColors.primaryBrown, fontSize: 16)),
          ),
          Switch(
            value: value,
            activeColor: AppColors.primaryBrown,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  void _showThemePicker() async {
    final mode = await showModalBottomSheet<AppThemeMode>(
      context: context,
      builder: (context) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(title: const Text('System'), onTap: () => Navigator.pop(context, AppThemeMode.system)),
              ListTile(title: const Text('Light'), onTap: () => Navigator.pop(context, AppThemeMode.light)),
              ListTile(title: const Text('Dark'), onTap: () => Navigator.pop(context, AppThemeMode.dark)),
            ],
          ),
        );
      },
    );
    if (mode == null) return;
    await SettingsService.setThemeMode(mode);
    setState(() => _theme = mode);
    // Notify root via navigator route arg callback if available
    final state = context.findAncestorStateOfType<State>();
    // Root wired via route: '/main' with onThemeChanged; we can trigger by
    // simply reopening MainNavigationPage which will read saved preference.
  }

  Future<void> _openHelpCenter() async {
    final uri = Uri.parse('https://github.com/raulsp/BeanScan-Application#readme');
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) return;
  }

  void _showAbout() {
    showAboutDialog(
      context: context,
      applicationName: 'BeanScan',
      applicationVersion: '1.0.0',
      applicationIcon: const Icon(Icons.coffee, color: AppColors.primaryBrown),
      children: const [
        Text('BeanScan helps analyze coffee bean images and detect defects.'),
      ],
    );
  }
} 