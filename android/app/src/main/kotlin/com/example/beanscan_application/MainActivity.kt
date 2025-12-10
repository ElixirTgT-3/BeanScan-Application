package com.example.beanscan_application

import android.view.KeyEvent
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private var volumePlugin: VolumeButtonsPlugin? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val plugin = VolumeButtonsPlugin()
        flutterEngine.plugins.add(plugin)
        volumePlugin = plugin
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        volumePlugin?.handleKeyEvent(event.keyCode, event)
        return super.dispatchKeyEvent(event)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        volumePlugin?.handleKeyEvent(keyCode, event)
        return super.onKeyDown(keyCode, event)
    }
}
