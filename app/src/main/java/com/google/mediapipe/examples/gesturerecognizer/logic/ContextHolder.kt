package com.google.mediapipe.examples.gesturerecognizer.logic

import android.util.Log
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult

object ContextHolder {
    private const val MAX_CAPACITY = 30
    private const val CONTEXT_HOLDER_TAG = "ContextHolder"
    private val dynamicSignCandidates
        = listOf("A", "C", "CZ", "D", "E", "F", "G", "H", "I", "K", "L", "N", "O", "R", "S", "SZ", "Z")
    private val labelsArray = mutableListOf<String>()
    var currentWord: String = ""

    fun appendLetterToCurrentWord(label: String?) {
        label?.let { it ->
            labelsArray.add(it)
            if (labelsArray.size == MAX_CAPACITY) {
                val halfArraySize = labelsArray.size / 2
                val mostCommonLabel = labelsArray
                    .groupingBy { it }
                    .eachCount()
                    .filter { it.value >= halfArraySize }
                    .maxByOrNull { it.value }?.key
                labelsArray.clear()
                mostCommonLabel?.takeIf { it.isNotBlank() }?.let { nonBlankLabel ->
                    currentWord += nonBlankLabel
                }
                labelsArray.clear()
            }
        }
    }

    private fun checkIfDynamic(label: String): Boolean {
        return label in dynamicSignCandidates
    }

    fun addGestureResult(result: GestureRecognizerResult) {
        if (result.gestures().isNotEmpty()) {
            val gesture = Gesture(result.gestures(), result.landmarks())
            if (checkIfDynamic(gesture.getCategory()))
                Log.i(CONTEXT_HOLDER_TAG, "possible dynamic sign")

            Log.i(CONTEXT_HOLDER_TAG, gesture.getCategory())
        }
    }
}

data class Gesture(
    val gesture: MutableList<MutableList<Category>>,
    val landmarks: MutableList<MutableList<NormalizedLandmark>>
) {
    fun getCategory(): String {
        return gesture[0][0].categoryName()
    }
    fun getLandmarkArray(): List<NormalizedLandmark> {
        return landmarks[0]
    }
}