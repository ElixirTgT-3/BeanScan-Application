import 'package:flutter/material.dart';

class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    final currentYear = DateTime.now().year;
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final background = theme.scaffoldBackgroundColor;
    final isDark = colorScheme.brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: background,
      appBar: AppBar(
        title: Text(
          'About BeanScan',
          style: theme.textTheme.titleMedium?.copyWith(
            color: colorScheme.primary,
            fontWeight: FontWeight.w600,
          ),
        ),
        backgroundColor: background,
        foregroundColor: colorScheme.primary,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          _HeroHeader(
            currentYear: currentYear,
            colorScheme: colorScheme,
            isDark: isDark,
          ),
          const SizedBox(height: 24),
          _InfoTile(
            icon: Icons.assistant_photo_outlined,
            title: 'Mission',
            body:
                'BeanScan helps coffee professionals classify bean varieties, flag defects, and estimate shelf life in seconds using a mobile device.',
            colorScheme: colorScheme,
            isDark: isDark,
          ),
          const SizedBox(height: 16),
          _InfoTile(
            icon: Icons.emoji_objects_outlined,
            title: 'Tech Stack',
            body:
                '- Flutter mobile app with adaptive UI.\n'
                '- PyTorch models for classification and detection.\n'
                '- FastAPI backend paired with Supabase history storage.',
            colorScheme: colorScheme,
            isDark: isDark,
          ),
          const SizedBox(height: 16),
          _InfoTile(
            icon: Icons.people_outline,
            title: 'Community',
            body:
                'Thousands of roasters and graders rely on BeanScan to keep shipments consistent and reduce manual inspection time.',
            colorScheme: colorScheme,
            isDark: isDark,
          ),
          const SizedBox(height: 24),
          _ContactCard(colorScheme: colorScheme),
        ],
      ),
    );
  }
}

class _HeroHeader extends StatelessWidget {
  final int currentYear;
  final ColorScheme colorScheme;
  final bool isDark;

  const _HeroHeader({
    required this.currentYear,
    required this.colorScheme,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    final primary = colorScheme.primary;
    final subtitleColor = colorScheme.onSurfaceVariant;
    final shadowColor = isDark
        ? Colors.black.withValues(alpha: 0.35)
        : primary.withValues(alpha: 0.12);

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: shadowColor,
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: primary.withValues(alpha: isDark ? 0.25 : 0.12),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Icon(
              Icons.local_cafe_outlined,
              color: primary,
              size: 32,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'BeanScan',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w800,
                    color: primary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Version 1.0.0 (c) $currentYear BeanScan Labs',
                  style: TextStyle(
                    color: subtitleColor,
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

class _InfoTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;
  final ColorScheme colorScheme;
  final bool isDark;

  const _InfoTile({
    required this.icon,
    required this.title,
    required this.body,
    required this.colorScheme,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    final primary = colorScheme.primary;
    final subtitleColor = colorScheme.onSurfaceVariant;
    final borderColor = colorScheme.outline.withValues(alpha: isDark ? 0.45 : 0.25);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: borderColor),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: primary.withValues(alpha: isDark ? 0.25 : 0.12),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(
              icon,
              color: primary,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 17,
                    color: primary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  body,
                  style: TextStyle(
                    color: subtitleColor,
                    height: 1.5,
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

class _ContactCard extends StatelessWidget {
  final ColorScheme colorScheme;

  const _ContactCard({required this.colorScheme});

  @override
  Widget build(BuildContext context) {
    final gradientStart = colorScheme.primary;
    final gradientEnd = colorScheme.primary.withValues(alpha: 0.8);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [gradientStart, gradientEnd],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Icon(
              Icons.email_outlined,
              color: Colors.white,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  "Let's Stay Connected",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                SizedBox(height: 6),
                Text(
                  'Email: hello@beanscan.app\nWebsite: www.beanscan.app\nTwitter/X: @beanscan',
                  style: TextStyle(
                    color: Colors.white,
                    height: 1.4,
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
