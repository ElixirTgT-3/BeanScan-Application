import 'package:flutter/material.dart';
import '../utils/app_colors.dart';

class HelpCenterPage extends StatelessWidget {
  const HelpCenterPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.lightBeige,
      appBar: AppBar(
        title: const Text('Help Center'),
        backgroundColor: AppColors.lightBeige,
        foregroundColor: AppColors.primaryBrown,
        elevation: 0,
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final horizontalPadding = constraints.maxWidth > 600 ? 48.0 : 24.0;
          final verticalPadding = constraints.maxWidth > 600 ? 28.0 : 20.0;
          return ListView.separated(
            padding: EdgeInsets.symmetric(
              horizontal: horizontalPadding,
              vertical: verticalPadding,
            ),
            separatorBuilder: (_, __) => const SizedBox(height: 18),
            itemCount: _helpEntries.length + 1,
            itemBuilder: (context, index) {
              if (index == 0) {
                return const _HeroCard();
              }
              final entry = _helpEntries[index - 1];
              return _HelpSection(
                icon: entry.icon,
                title: entry.title,
                bulletPoints: entry.bulletPoints,
              );
            },
          );
        },
      ),
    );
  }
}

class _HelpEntry {
  final IconData icon;
  final String title;
  final List<String> bulletPoints;

  const _HelpEntry({
    required this.icon,
    required this.title,
    required this.bulletPoints,
  });
}

const List<_HelpEntry> _helpEntries = [
  _HelpEntry(
    icon: Icons.play_circle_outline,
    title: 'Getting Started',
    bulletPoints: [
      'Fill the camera frame with coffee beans on a neutral background.',
      'Use bright, even lighting to minimise shadows or glare.',
      'Tap the shutter to analyse; auto-save keeps a history entry by default.',
    ],
  ),
  _HelpEntry(
    icon: Icons.camera_alt_outlined,
    title: 'Live Scan Tips',
    bulletPoints: [
      'Toggle flash when working in dim rooms or warehouses.',
      'Keep the phone steady—rest elbows or use a stand for crisp captures.',
      'Use the gallery picker for pre-shot batches or archived images.',
    ],
  ),
  _HelpEntry(
    icon: Icons.insights_outlined,
    title: 'Understanding Results',
    bulletPoints: [
      'Confidence Score shows how sure the model is about the bean type.',
      'Estimated Months updates according to detected defects and severity.',
      'Tap “No” on the results page to jump back to history without rescanning.',
    ],
  ),
  _HelpEntry(
    icon: Icons.handyman_outlined,
    title: 'Troubleshooting',
    bulletPoints: [
      'Clean the camera lens and refocus if the preview looks hazy.',
      'Retake from different angles when beans overlap heavily.',
      'Check your network connection to keep the API responsive.',
      'Restart the app if a scan freezes or runs longer than usual.',
    ],
  ),
  _HelpEntry(
    icon: Icons.history_edu_outlined,
    title: 'Managing History',
    bulletPoints: [
      'Recent scans appear instantly in the History tab.',
      'Tap any history row to reopen full bean and defect details.',
      'Toggle auto-save in Settings under “General Preferences”.',
    ],
  ),
  _HelpEntry(
    icon: Icons.settings_outlined,
    title: 'Settings Overview',
    bulletPoints: [
      'Enable or disable auto-save to control when scans are stored.',
      'Switch between Light and Dark themes to match your environment.',
      'Access Help Center and About pages for documentation and credits.',
    ],
  ),
  _HelpEntry(
    icon: Icons.support_agent_outlined,
    title: 'More Assistance',
    bulletPoints: [
      'Email: support@beanscan.app',
      'Forum: forum.beanscan.app',
      'Documentation: docs.beanscan.app',
    ],
  ),
];

class _HeroCard extends StatelessWidget {
  const _HeroCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.brown.withOpacity(0.08),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: AppColors.primaryBrown.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Icon(
              Icons.coffee_outlined,
              color: AppColors.primaryBrown,
              size: 32,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Your Bean Coach',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: AppColors.primaryBrown,
                  ),
                ),
                SizedBox(height: 6),
                Text(
                  'Find quick tips, fixes, and contact options to keep scanning smoothly.',
                  style: TextStyle(
                    color: AppColors.textGrey,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HelpSection extends StatelessWidget {
  final IconData icon;
  final String title;
  final List<String> bulletPoints;

  const _HelpSection({
    required this.icon,
    required this.title,
    required this.bulletPoints,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.dividerGrey.withOpacity(0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: AppColors.primaryBrown.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  color: AppColors.primaryBrown,
                  size: 22,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 17,
                        color: AppColors.primaryBrown,
                      ),
                    ),
                    const SizedBox(height: 12),
                    ..._buildBulletList(),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  List<Widget> _buildBulletList() {
    return bulletPoints
        .map(
          (point) => Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Padding(
                  padding: EdgeInsets.only(top: 4),
                  child: Icon(
                    Icons.circle,
                    size: 6,
                    color: AppColors.primaryBrown,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    point,
                    style: const TextStyle(
                      color: AppColors.textDarkGrey,
                      height: 1.5,
                    ),
                  ),
                ),
              ],
            ),
          ),
        )
        .toList();
  }
}
