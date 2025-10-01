import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

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
                _buildSettingItem(Icons.download, "Auto-Save Scans"),
                _buildDivider(),
                _buildSettingItem(Icons.dark_mode, "Theme"),
              ],
            ),
            _buildSection(
              title: "Support & About",
              children: [
                _buildSettingItem(Icons.help_outline, "Help Center"),
                _buildDivider(),
                _buildSettingItem(Icons.info_outline, "About the App"),
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

  Widget _buildSettingItem(IconData icon, String title) {
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
          const Icon(
            Icons.chevron_right,
            color: AppColors.primaryBrown,
          ),
        ],
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
} 