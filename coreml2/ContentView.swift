//
//  ContentView.swift
//  coreml2
//
//  Created by Yeddula Nachiketh Reddy on 26/10/25.
//

import CoreML
import Vision
import UIKit
import SwiftUI
import Combine

// MARK: - ContentView (Main UI)
struct ContentView: View {
    @StateObject private var modelView = ModelView()
    @State private var selectedImage: UIImage?
    @State private var isImagePickerPresented = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // App Title
                Text("Fruit Classifier")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding(.top)
                
                // Show selected image or placeholder
                ZStack {
                    RoundedRectangle(cornerRadius: 15)
                        .fill(Color.gray.opacity(0.1))
                        .frame(height: 300)
                        .overlay(
                            RoundedRectangle(cornerRadius: 15)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 2)
                        )
                    
                    if let selectedImage = selectedImage {
                        Image(uiImage: selectedImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 280)
                            .cornerRadius(10)
                    } else {
                        VStack {
                            Image(systemName: "photo")
                                .font(.system(size: 60))
                                .foregroundColor(.gray)
                            Text("Select an image")
                                .foregroundColor(.gray)
                                .padding(.top, 5)
                        }
                    }
                }
                .padding(.horizontal)
                
                // Display the classification result
                VStack(spacing: 10) {
                    if modelView.isProcessing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(1.5)
                    } else {
                        Text(modelView.classificationLabel)
                            .font(.title2)
                            .fontWeight(.semibold)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.primary)
                        
                        if let confidence = modelView.confidence {
                            Text("Confidence: \(confidence)%")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .frame(height: 80)
                .padding()
                
                Spacer()
                
                // Button to open the image picker
                Button(action: {
                    isImagePickerPresented = true
                }) {
                    HStack {
                        Image(systemName: "photo.fill")
                        Text("Choose Image")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(width: 250, height: 55)
                    .background(
                        LinearGradient(
                            gradient: Gradient(colors: [Color.blue, Color.blue.opacity(0.8)]),
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
                    .shadow(radius: 5)
                }
                .padding(.bottom, 30)
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $isImagePickerPresented) {
                ImagePicker(selectedImage: $selectedImage) { newImage in
                    modelView.classify(image: newImage)
                }
            }
        }
    }
}

// MARK: - ImagePicker
struct ImagePicker: UIViewControllerRepresentable {
    @Environment(\.presentationMode) private var presentationMode
    @Binding var selectedImage: UIImage?
    var completion: (UIImage) -> Void
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        picker.allowsEditing = false
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // No updates needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]
        ) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
                parent.completion(image)
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - ModelView (ViewModel for Classification)
//code will go here

class ModelView: ObservableObject {
    @Published var classificationLabel: String = "No image selected"
    @Published var confidence: Int?
    @Published var isProcessing: Bool = false
    
    private lazy var classificationRequest: VNCoreMLRequest = {
        do {
            let configuration = MLModelConfiguration()
#if targetEnvironment(simulator)
            configuration.computeUnits = .cpuOnly
#else
            configuration.computeUnits = .all
#endif
            let model = try FruitClassifier(configuration: configuration)
            let visionModel = try VNCoreMLModel(for: model.model)
            let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
                self?.processClassification(for: request, error: error)
            }
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Core ML model: \(error)")
        }
    }()
    func classify(image: UIImage){
        guard let cgimage = image.cgImage else {
            self.classificationLabel = "Failed to process image"
            return
        }
        self.isProcessing = true
        self.classificationLabel = "Processing..."
        self.confidence = nil
        
        let handler = VNImageRequestHandler(cgImage: cgimage, orientation: .up, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("Failed to perform classification. Error: \(error)")
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.classificationLabel = "Failed to process image"
                }
            }
        }
        
    }
    private func processClassification(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            self.isProcessing = false
            
            if let error = error {
                self.classificationLabel = "Classification failed: \(error.localizedDescription)"
                self.confidence = nil
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                self.classificationLabel = "No results found"
                self.confidence = nil
                return
            }
            
            let confidence = Int(topResult.confidence * 100)
            let fruitName = topResult.identifier.capitalized
            let fruitEmoji = self.getEmoji(for: fruitName)
            
            self.classificationLabel = "\(fruitEmoji) \(fruitName)"
            self.confidence = confidence
        }
    }
    
    private func getEmoji(for fruitName: String) -> String {
        let fruit = fruitName.lowercased()
        
        switch fruit {
        case let name where name.contains("apple"): return "🍎"
        case let name where name.contains("banana"): return "🍌"
        case let name where name.contains("pineapple"): return "🍍"
        default : return "🍎"
        }
    }
}
    



// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

#Preview {
    ContentView()
}
