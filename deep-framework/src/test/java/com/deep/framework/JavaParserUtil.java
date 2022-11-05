package com.deep.framework;

import com.deep.framework.framework.TensorCompiler;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.graph.TensorFunctor;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.printer.YamlPrinter;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ClassLoaderTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.ParserCollectionStrategy;
import com.github.javaparser.utils.ProjectRoot;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class JavaParserUtil {

    private static AtomicInteger id = new AtomicInteger();
    private static Map<String, TensorFunctor> mapf = new HashMap<>();
    private static Map<Node, Integer> mapx = new HashMap<>();

    static {
        TensorCompiler tc = new TensorCompiler();
        Method[] methods = tc.getClass().getDeclaredMethods();
        Arrays.stream(methods).forEach(method -> {
            try {
                Class type = (Class) method.getGenericParameterTypes()[0];
                Tensor[] args = IntStream.range(0, method.getParameterCount()).mapToObj(a -> new Tensor(new int[]{1})).toArray(Tensor[]::new);
                TensorFunctor tensor = (TensorFunctor) method.invoke(tc, type.isArray() ? new Object[]{args} : args);
                mapf.put(method.getName(), tensor);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public static void yamlPrinter(Node node) {
        YamlPrinter yamlPrinter = new YamlPrinter(true);
        System.out.println(yamlPrinter.output(node));
    }

    private static JavaSymbolSolver getJavaSymbolSolver(String path) {
        CombinedTypeSolver solver = new CombinedTypeSolver();
        solver.add(new JavaParserTypeSolver(path));
        solver.add(new ReflectionTypeSolver());
        solver.add(new ClassLoaderTypeSolver(JavaParserUtil.class.getClassLoader()));
        return new JavaSymbolSolver(solver);
    }

    public static void parseProject(String path) {
        Path root = Paths.get(path);
        ProjectRoot projectRoot = new ParserCollectionStrategy().collect(root);
        projectRoot.getSourceRoots().forEach(sourceRoot -> {
            JavaSymbolSolver symbolSolver = getJavaSymbolSolver(path);
            sourceRoot.getParserConfiguration().setSymbolResolver(symbolSolver);
            try {
                sourceRoot.tryToParse();
            } catch (IOException e) {
                e.printStackTrace();
            }

            sourceRoot.getCompilationUnits().forEach(cu -> {
                cu.getClassByName("TensorFlow").ifPresent(o -> o.getMembers().stream().map(BodyDeclaration::asMethodDeclaration).forEach(a -> {
                    if (a.getName().asString().equals("softmax")) {
                        a.getBody().get().getStatements().getFirst().get().asReturnStmt().getExpression().get().asObjectCreationExpr().getAnonymousClassBody().get().forEach(b -> {
                            b.asMethodDeclaration().getBody().get().getChildNodes().forEach(JavaParserUtil::forwards);
                            b.asMethodDeclaration().getBody().get().getChildNodes().forEach(JavaParserUtil::backwards);
                        });
                    }
                }));
            });
        });
    }

    public static void forwards(Node node) {
        for (Node n : node.getChildNodes()) {
            forwards(n);
        }
        forward(node);
    }

    public static void backwards(Node node) {
        backward(node);
        for (Node n : node.getChildNodes()) {
            backwards(n);
        }
    }

    public static void forward(Node node) {
        if (node instanceof MethodCallExpr expression) {
            if (expression.isMethodCallExpr()) {
                MethodCallExpr methodCallExpr = expression.asMethodCallExpr();
                TensorFunctor functor = mapf.get(methodCallExpr.getName().asString());

                if (Objects.nonNull(functor)) {


                    id.getAndIncrement();
                    mapx.put(methodCallExpr, id.get());

                    Tensor[] tensors = getTensors(methodCallExpr);
                    functor.setId(mapx.get(methodCallExpr));
                    functor.setInput(tensors);
                    functor.compute();
                    return;
                }

                if (methodCallExpr.getName().getIdentifier().equals("forEach")) {
                    System.out.println("forEach");
                }
            } else if (expression.isNameExpr()) {
                NameExpr nameExpr = expression.asNameExpr();
                nameExpr.getName().asString().concat(";");
            }
        }
    }

    public static void backward(Node node) {
        if (node instanceof Expression expression) {
            if (expression.isMethodCallExpr()) {
                MethodCallExpr methodCallExpr = expression.asMethodCallExpr();
                TensorFunctor functor = mapf.get(methodCallExpr.getName().asString());

                if (Objects.nonNull(functor)) {
                    Tensor[] tensors = getTensors(methodCallExpr);
                    functor.setId(mapx.get(methodCallExpr));
                    functor.setInput(tensors);
                    functor.gradient("");
                    return;
                }

                if (methodCallExpr.getName().getIdentifier().equals("forEach")) {
                    System.out.println("forEach");
                }

            } else if (expression.isNameExpr()) {
                NameExpr nameExpr = expression.asNameExpr();
                nameExpr.getName().asString().concat(";");
            }
        }
    }

    private static Tensor[] getTensors(MethodCallExpr methodCallExpr) {

//        CombinedTypeSolver solver = new CombinedTypeSolver();
//        solver.add(new JavaParserTypeSolver("/Users/chengdong/GitHub/deep-learning/deep-framework/src/main/"));
//        solver.add(new ReflectionTypeSolver());
//
//        JavaParserFacade javaParserFacade = JavaParserFacade.get(solver);
//        SymbolReference<ResolvedMethodDeclaration> resolvedMethodDeclarationSymbolReference = javaParserFacade.solve(methodCallExpr);
//        System.out.println("is resolved: " + resolvedMethodDeclarationSymbolReference.isSolved());
//        System.out.println("resolved type" + resolvedMethodDeclarationSymbolReference.getCorrespondingDeclaration());

        ResolvedMethodDeclaration resolve = methodCallExpr.resolve();
        System.out.println(resolve.getReturnType());

        Tensor[] tensors = methodCallExpr.getArguments().stream().map(a -> {
            if (a.isObjectCreationExpr()) {
                ObjectCreationExpr objectCreationExpr = a.asObjectCreationExpr();
                if (objectCreationExpr.getType().asString().equals("TensorConst")) {
                    String value = objectCreationExpr.getArguments()
                    .stream().filter(Expression::isDoubleLiteralExpr)
                    .map(c -> c.asDoubleLiteralExpr().getValue())
                    .collect(Collectors.joining(","));
                    return new TensorConst(Double.parseDouble(value));
                }
                return new Tensor(0);
            }
            Tensor tensor = new Tensor(0);
            None output = tensor.getOutput();
            output.setId(mapx.get(a));
            return tensor;
        }).toArray(Tensor[]::new);
        return tensors;
    }

    public static void main(String[] args) {
        parseProject("GitHub/deep-learning/deep-framework/src/main/");
    }

}
