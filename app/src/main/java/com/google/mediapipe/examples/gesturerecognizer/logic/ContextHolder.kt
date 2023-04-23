package com.google.mediapipe.examples.gesturerecognizer.logic

import android.util.Log
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark


object ContextHolder {
    private const val MAX_CAPACITY = 20
    private const val GESTURE_MAX_CAPACITY = 4
    private const val MOVEMENT_THRESHOLD = 0.2
    private const val CONTEXT_HOLDER_TAG = "ContextHolder"
    private val dynamicSignCandidates = listOf("A", "C", "CZ", "D", "E", "F", "G", "H", "I", "K", "L", "N", "O", "R", "S", "SZ", "Z")
    private val labelsArray = mutableListOf<String>()
    private val gesturesArray = ArrayDeque<Gesture>()
    var currentWord: String = ""

    fun addGestureResult(result: GestureRecognizerResult) {
        Log.i(CONTEXT_HOLDER_TAG, "$gesturesArray")
        if (result.gestures().isNotEmpty()) {
            val gesture = Gesture(result.gestures(), result.landmarks())
            addGesture(gesture)
            Log.i(CONTEXT_HOLDER_TAG, gesture.getCategory())

            if (!isDynamic(gesture.getCategory())) {
                Log.i(CONTEXT_HOLDER_TAG, "static sign")
                gesturesArray.clear()
            } else {
                Log.i(CONTEXT_HOLDER_TAG, "possible dynamic sign")
                if (doesMatchCurrent(gesture) && gesturesArray.isNotEmpty()) {
                    Log.i(CONTEXT_HOLDER_TAG, "current sign matches most")
                    comparePositions(
                        gesturesArray.first().getLandmarkArray(),
                        gesturesArray.last().getLandmarkArray(),
                        HandLandmark.WRIST
                    )
                }
            }
            appendLetterToCurrentWord(gesture.getCategory())
        }
    }

    private fun comparePositions(
        first: List<NormalizedLandmark>,
        last: List<NormalizedLandmark>,
        index: Int
    ) {
        // todo: tutaj dodać sensowny threshold
        if (first[index].y() > last[index].y()) {
            Log.i(CONTEXT_HOLDER_TAG, "WRIST MOVED DOWN")
        } else {
            Log.i(CONTEXT_HOLDER_TAG, "WRIST MOVED UP")
        }
    }

    // czy to jest potrzebne ???
    private fun doesMatchCurrent(candidate: Gesture): Boolean {
        val count = gesturesArray.count { it.getCategory() == candidate.getCategory() }
        return count > gesturesArray.size - (GESTURE_MAX_CAPACITY / 2)
    }

    private fun isDynamic(label: String): Boolean {
        return label in dynamicSignCandidates
    }

    private fun addGesture(gesture: Gesture) {
        gesturesArray.addFirst(gesture)
        if (gesturesArray.size > GESTURE_MAX_CAPACITY) {
            gesturesArray.removeLast()
        }
        if (gesturesArray.none { it.getCategory() == gesture.getCategory() }) {
            gesturesArray.clear()
        }
    }

    private fun appendLetterToCurrentWord(label: String) {
        labelsArray.add(label)
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