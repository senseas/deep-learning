package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.expression.TypeParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.deep.framework.ast.lexer.TokenType.*;

@Data
public class ClassOrInterfaceDeclaration extends TypeDeclaration {
    private List<Object> extendedTypes;
    private List<Object> implementedTypes;
    private List<TokenType> modifiers;
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;

    public ClassOrInterfaceDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        AnnotationDeclaration.parser(node);
        PackageDeclaration.parser(node);
        ImportDeclaration.parser(node);

        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (b instanceof BlockStatement) {
                Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                    if (m.equals(CLASS)) {
                        ClassOrInterfaceDeclaration declare = new ClassOrInterfaceDeclaration(node.getPrarent());
                        List<TokenType> modifiers = a.getFieldModifiers();

                        b.setPrarent(declare);
                        declare.setModifiers(modifiers);
                        declare.setName((Name) n);
                        declare.setBody((BlockStatement) b);
                        declare.setChildrens(a.getChildrens());
                        declare.getChildrens().add(b);
                        a.getChildrens().remove(m);
                        node.replaceAndRemove(a, declare, b);

                        TypeParametersExpression.parser(declare);
                        parserImplements(declare);
                        parserExtends(declare);

                        ConstructorDeclaration.parser(b);
                        MethodDeclaration.parser(b);
                        FieldDeclaration.parser(b);
                    }
                });
            }
        });
    }

    public static void parserImplements(ClassOrInterfaceDeclaration classDeclare) {
        if (classDeclare.getChildrens().contains(IMPLEMENTS)) {
            List<Object> list = null;
            for (Object a : List.copyOf(classDeclare.getChildrens())) {
                if (a instanceof BlockStatement) {
                    classDeclare.setImplementedTypes(list);
                    classDeclare.getChildrens().removeAll(list);
                    return;
                } else if (a.equals(IMPLEMENTS)) {
                    list = new ArrayList<>();
                    classDeclare.getChildrens().remove(a);
                } else if (Objects.nonNull(list)) {
                    list.add(a);
                }
            }
        }
    }

    public static void parserExtends(ClassOrInterfaceDeclaration classDeclare) {
        if (classDeclare.getChildrens().contains(EXTENDS)) {
            List<Object> list = null;
            for (Object a : List.copyOf(classDeclare.getChildrens())) {
                if (a.equals(IMPLEMENTS)) {
                    classDeclare.setExtendedTypes(list);
                    classDeclare.getChildrens().removeAll(list);
                    return;
                } else if (a instanceof BlockStatement) {
                    classDeclare.setExtendedTypes(list);
                    classDeclare.getChildrens().removeAll(list);
                    return;
                } else if (a.equals(EXTENDS)) {
                    list = new ArrayList<>();
                    classDeclare.getChildrens().remove(a);
                } else if (Objects.nonNull(list)) {
                    list.add(a);
                }
            }
        }
    }
}