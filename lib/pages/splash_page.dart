import 'package:flutter/material.dart';
import '../main.dart';

class SplashPage extends StatefulWidget {
  const SplashPage({super.key});

  @override
  State<SplashPage> createState() => _SplashPageState();
}

class _SplashPageState extends State<SplashPage> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 1400), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const MainNavigationPage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bool isDark = theme.brightness == Brightness.dark;
    final String backgroundImage = isDark
        ? 'https://storage.googleapis.com/tagjs-prod.appspot.com/v1/ovGha4FhsH/y9nv6az7_expires_30_days.png'
        : 'https://storage.googleapis.com/tagjs-prod.appspot.com/v1/ovGha4FhsH/4f9rxn80_expires_30_days.png';

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Container(
          width: double.infinity,
          height: double.infinity,
          decoration: BoxDecoration(
            image: DecorationImage(
              image: NetworkImage(backgroundImage),
              fit: BoxFit.cover,
            ),
          ),
        ),
      ),
    );
  }
}


